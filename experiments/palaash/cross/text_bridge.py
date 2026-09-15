"""Cross-family "stitch": Llama-3.2-3B-Instruct (large) -> Qwen2.5-0.5B-Instruct
(small), via a TEXT bridge rather than activation injection.

Why not activation stitching
-----------------------------
Every other package in this repo (`stitching_large_to_small`, its sibling, and
`diagnosis`'s align="token" pairs) works by teacher-forcing ONE token sequence
through both models and splicing hidden states position-by-position -- which
only makes sense because Llama-1B/3B and Qwen-0.5B/3B each share a tokenizer
within their family. `stitching_large_to_small/config.py` says so explicitly:
"Token-aligned pairs only ... Cross-family is out of scope for v1." Llama and
Qwen do not share a vocabulary, so Llama token ids fed to Qwen's embedding
table (or vice versa) address completely unrelated tokens -- there is no
position at which "splice the residual here" is a coherent operation, and the
runner's own per-step decode would have to feed the same generated token id to
both models' embeddings every step, which is equally broken across tokenizers.

So this experiment reads "stitching" at the text level instead of the
activation level: the large model's own greedy answer is generated first, then
handed to the small model as an unverified suggestion inside its prompt, and
the small model produces the final, scored answer on its own. This is a much
weaker coupling than activation injection -- it is prompting, not
representation transplantation -- but it is the only version of "large model's
knowledge, small model's forward pass" that is architecturally sound when the
two models do not share a tokenizer.

Design
------
    large.generate(question)              -> hint text (short, same budget as
                                               this bank's teacher answer)
    small.generate(question + hint)       -> final, scored answer

Latency is the SUM of both generations (the large model's hint pass is not
free), timed through the same KV-cached `common.decoding` loop every other
baseline and stitch in this repo uses, so the comparison is apples-to-apples
with `temporary/`'s numbers in methodology if not in magnitude.

Bank, split, and models
------------------------
`hard_factual` (422 obscure short-answer questions), same bank
`stitching_large_to_small` is built around. No adapter is fit here -- there is
nothing to train, so there is no fit/eval split to keep disjoint, and the
FULL bank is scored (unlike the temporary/ scripts, which reserve half of it
for fitting). Models are the pair diagnosis/config.py already names
`qwen2llama`: Llama-3.2-3B-Instruct (large) and Qwen2.5-0.5B-Instruct (small).

Writes results_llama_to_qwen_bridge.csv in this folder.

    python text_bridge.py               # full run (422 prompts)
    python text_bridge.py --smoke       # tiny smoke test first
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PALAASH_ROOT = HERE.parent
sys.path.insert(0, str(PALAASH_ROOT))

from common.decoding import FullRunner, eos_ids, greedy, warmup  # noqa: E402
from common.model_utils import build_prompt_ids, load_lm, pick_device  # noqa: E402
from common.scoring import score  # noqa: E402
from stitching_large_to_small.config import BANKS  # noqa: E402
from stitching_large_to_small.data import split_prompts  # noqa: E402

LARGE_ID, LARGE_TAG = "meta-llama/Llama-3.2-3B-Instruct", "Llama-3B"
SMALL_ID, SMALL_TAG = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen-0.5B"
BANK = BANKS["hard_factual"]
HINT_TOKENS = BANK.teacher_answer_tokens   # 16: the large model's own short answer
ANSWER_TOKENS = BANK.max_new_tokens        # 24: what every scored generation uses

OUT_CSV = HERE / "results_llama_to_qwen_bridge.csv"


def hybrid_question(question: str, hint: str) -> str:
    """Splice the large model's greedy answer into the small model's prompt as
    an unverified suggestion. The hint is not guaranteed correct (the large
    model is ~97% on this bank, not 100%), so the small model is told it may be
    wrong rather than instructed to just repeat it -- copying a wrong hint
    verbatim would understate what the small model itself contributes."""
    hint = hint.strip()
    if not hint:
        return question
    return (f'{question}\n\n(A colleague suggested the answer might be '
            f'"{hint}" -- they may be wrong. Give your own best final answer.)')


@torch.no_grad()
def run_condition(label: str, prompts: list[dict], answer_fn) -> dict:
    """answer_fn(item) -> (generated_text, total_ms). Shared accuracy/latency
    bookkeeping for all three conditions (large alone / small alone / hybrid)."""
    rows, total_ms = {}, 0.0
    for k, p in enumerate(prompts):
        text, ms = answer_fn(p)
        rows[p["id"]] = {"correct": bool(score(text, p)), "text": text}
        total_ms += ms
        if k % 50 == 0 or k == len(prompts) - 1:
            print(f"  [{label}] {k + 1}/{len(prompts)}", flush=True)
    n = len(prompts)
    n_ok = sum(r["correct"] for r in rows.values())
    return {"label": label, "n": n, "accuracy": n_ok / n if n else float("nan"),
            "mean_ms_per_answer": total_ms / n if n else float("nan"), "rows": rows}


def divergent_ids(small_rows: dict, large_rows: dict) -> list[str]:
    return sorted(pid for pid in small_rows
                  if not small_rows[pid]["correct"] and large_rows.get(pid, {}).get("correct"))


def subset_acc(rows: dict, ids: list[str]) -> float:
    if not ids:
        return float("nan")
    return sum(rows[i]["correct"] for i in ids) / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run (5 prompts) to check correctness")
    a = ap.parse_args()

    device = pick_device()
    print(f"[main] device={device}  large={LARGE_ID}  small={SMALL_ID}  smoke={a.smoke}")
    lm_large = load_lm(LARGE_ID, LARGE_TAG, device)
    lm_small = load_lm(SMALL_ID, SMALL_TAG, device)
    stop_large, stop_small = eos_ids(lm_large), eos_ids(lm_small)

    prompts = split_prompts(BANK, "all")
    if a.smoke:
        prompts = prompts[:5]
    print(f"[data] {len(prompts)} {BANK.name} prompts (full bank; nothing is fit here)")

    runner_large = FullRunner(lm_large, LARGE_TAG)
    runner_small = FullRunner(lm_small, SMALL_TAG)
    warmup(runner_large, build_prompt_ids(lm_large.tokenizer, prompts[0]["question"],
                                          device, BANK.system))
    warmup(runner_small, build_prompt_ids(lm_small.tokenizer, prompts[0]["question"],
                                          device, BANK.system))

    def large_alone(p: dict) -> tuple[str, float]:
        ids = build_prompt_ids(lm_large.tokenizer, p["question"], device, BANK.system)
        out = greedy(runner_large, ids, ANSWER_TOKENS, stop_large)
        return lm_large.tokenizer.decode(out["token_ids"], skip_special_tokens=True).strip(), out["total_ms"]

    def small_alone(p: dict) -> tuple[str, float]:
        ids = build_prompt_ids(lm_small.tokenizer, p["question"], device, BANK.system)
        out = greedy(runner_small, ids, ANSWER_TOKENS, stop_small)
        return lm_small.tokenizer.decode(out["token_ids"], skip_special_tokens=True).strip(), out["total_ms"]

    def hybrid(p: dict) -> tuple[str, float]:
        hint_ids = build_prompt_ids(lm_large.tokenizer, p["question"], device, BANK.system)
        hint_out = greedy(runner_large, hint_ids, HINT_TOKENS, stop_large)
        hint = lm_large.tokenizer.decode(hint_out["token_ids"], skip_special_tokens=True).strip()

        q2 = hybrid_question(p["question"], hint)
        final_ids = build_prompt_ids(lm_small.tokenizer, q2, device, BANK.system)
        final_out = greedy(runner_small, final_ids, ANSWER_TOKENS, stop_small)
        text = lm_small.tokenizer.decode(final_out["token_ids"], skip_special_tokens=True).strip()
        return text, hint_out["total_ms"] + final_out["total_ms"]

    print(f"\n[large] {LARGE_TAG} alone")
    large_res = run_condition(LARGE_TAG, prompts, large_alone)
    print(f"  acc={large_res['accuracy']:.1%}  ms/answer={large_res['mean_ms_per_answer']:.0f}")

    print(f"\n[small] {SMALL_TAG} alone")
    small_res = run_condition(SMALL_TAG, prompts, small_alone)
    print(f"  acc={small_res['accuracy']:.1%}  ms/answer={small_res['mean_ms_per_answer']:.0f}")

    print(f"\n[hybrid] {SMALL_TAG} + {LARGE_TAG} text hint")
    hybrid_res = run_condition("hybrid", prompts, hybrid)
    print(f"  acc={hybrid_res['accuracy']:.1%}  ms/answer={hybrid_res['mean_ms_per_answer']:.0f}")

    div_ids = divergent_ids(small_res["rows"], large_res["rows"])
    print(f"\n[divergent] {SMALL_TAG} wrong, {LARGE_TAG} right: {len(div_ids)} prompts")

    large_acc, large_ms = large_res["accuracy"], large_res["mean_ms_per_answer"]
    rows = []
    for label, res, notes in (
        (f"{LARGE_TAG}_alone", large_res, "reference (what the hint is graded against)"),
        (f"{SMALL_TAG}_alone", small_res, "cheap baseline, no large-model involvement"),
        (f"{SMALL_TAG}_plus_{LARGE_TAG}_hint", hybrid_res,
         "text-bridge hybrid: large generates a hint, small gives the scored answer"),
    ):
        rows.append({
            "label": label, "n_eval": res["n"], "accuracy": res["accuracy"],
            "accuracy_drop_vs_large_pts": (large_acc - res["accuracy"]) * 100,
            "divergent_accuracy": subset_acc(res["rows"], div_ids), "n_divergent": len(div_ids),
            "mean_ms_per_answer": res["mean_ms_per_answer"],
            "speedup_vs_large": large_ms / res["mean_ms_per_answer"] if res["mean_ms_per_answer"] else float("nan"),
            "notes": notes,
        })

    if not a.smoke:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[done] wrote {OUT_CSV}")
    else:
        print("\n[done] smoke test finished (no CSV written)")

    print("\n" + "=" * 88)
    print(f"{'condition':<32}{'accuracy':>10}{'drop vs large':>16}{'ms/answer':>12}{'speed vs large':>16}")
    for r in rows:
        print(f"{r['label']:<32}{r['accuracy']:>9.1%} {r['accuracy_drop_vs_large_pts']:>14.1f}pt "
              f"{r['mean_ms_per_answer']:>11.0f} {r['speedup_vs_large']:>14.2f}x")


if __name__ == "__main__":
    main()
