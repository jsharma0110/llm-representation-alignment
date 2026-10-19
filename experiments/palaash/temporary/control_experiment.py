"""The four-way control that actually tests "is this a stitching win", not
just "did accuracy go up": A (1B alone) / B (1B tail fine-tuned on the 3B
teacher, NO stitch) / C (frozen stitch, exit+add) / D (stitch + unfrozen
tail, exit+add -- the sibling package's existing 88.7% result).

Why exit, not warm
-------------------
Under `warm` the 1B model prefills the prompt itself, so the FIRST generated
token is always the 1B's own decision -- and on hard_factual, the answer is
often exactly that one token. A stitch can only ever "hover around 1B" under
warm (see the LoRA run in this folder's history: 82% vs 81%), because the 3B
signal never gets a chance to decide anything. `exit` stitches the prompt
too, so the adapted representation reaches the position that emits the first
answer token.

Why this needs a NEW control (B), not just A/C/D
--------------------------------------------------
D already exists on disk (stitching_large_to_small/results/.../
bench_i08_j14_dev_exit_distill_add_unfroze.json: 88.7% vs 70.8%, +17.9pts,
beats_small_ci=True) and C already exists too (bench_i08_j14_dev_exit_
distill_add.json: 72.6% vs 70.8%, CIs overlap, NOT a win) -- both read
read-only below, not regenerated. What does NOT exist anywhere is B: 1B with
blocks 8-15 fine-tuned on the SAME distillation signal (KL to the 3B
teacher's next-token distribution) on the SAME data, but with NO 3B
computation involved at inference at all -- no stitched prefix, no adapter.
B isolates whether D's gain came from the 3B's actual mid-layer
representations being used, or just from fine-tuning 1B's own tail on a
better training signal (which the sibling package's own architecture cannot
measure, since it always routes through an adapter+3B-prefix).

    C >> A            -> a real frozen-stitch win (contradicts what's on disk)
    D >> A but D ~= B  -> fine-tuning won; the 3B prefix wasn't doing the work
    D >> B             -> unfreezing helped AND the 3B mid-layers add something

Matched to D's exact recipe (read from adapter_i08_j14_distill_add_unfroze.json):
211 hard_factual bank fit-split items + 700 items from the local `generic`
corpus (common/fit_corpus.py, offline, seed=0) = 911 fit items, template
variety on, distill_epochs=4, distill_lr=2e-6, distill_tail_lr=2e-6,
distill_batch_seqs=4, train_mode="exit", injection_mode="add".

A/B/C are evaluated here, fresh, on the exact `dev` split (106 prompts) so
they are directly comparable to each other and to D's on-disk dev numbers.
The divergent set (1B wrong, 3B right) is computed fresh from THIS session's
own A run, not borrowed from a different run's baseline (bf16 on MPS is not
bit-deterministic across processes -- the sibling package's own README notes
this).

Only imports (never edits) the shared `common` utilities and the
geometry-agnostic pieces of the sibling `stitching_large_to_small` package.
D's numbers are READ from its existing bench file, never regenerated.
Writes results_control_ABCD.csv in this folder.

    python control_experiment.py               # full run
    python control_experiment.py --smoke        # tiny correctness check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PALAASH_ROOT = HERE.parent
sys.path.insert(0, str(PALAASH_ROOT))

from common.decoding import FullRunner, Stack, eos_ids, greedy  # noqa: E402
from common.fit_corpus import load_corpus  # noqa: E402
from common.model_utils import build_prompt_ids, load_lm, pick_device  # noqa: E402
from common.scoring import score  # noqa: E402
from common.templates import apply_framing, variant_for  # noqa: E402
from stitching_large_to_small.adapter import TorchAdapter, zero_map  # noqa: E402
from stitching_large_to_small.config import (  # noqa: E402
    BANKS, DISTILL_TOPK, PAIRS, PRESERVE_PREFIX,
)
from stitching_large_to_small.data import (  # noqa: E402
    layer_states, split_prompts, teacher_sequence, topk_teacher,
)
from stitching_large_to_small.distill import kl_to_teacher  # noqa: E402
from stitching_large_to_small.distill import train as distill_train  # noqa: E402
from stitching_large_to_small.stitch import LargeToSmallRunner  # noqa: E402

PAIR = PAIRS["llama"]
SMALL_ID, LARGE_ID = PAIR.small_id, PAIR.large_id
I_SMALL, J_LARGE = 8, 14
BANK = BANKS["hard_factual"]
INJECTION_MODE = "add"
TRAIN_MODE = "exit"
MODE = "exit"          # bench/decode mode -- must match TRAIN_MODE

N_CORPUS = 700          # matches D's exact recipe
EPOCHS = 4
LR = 2e-6
TAIL_LR = 2e-6
BATCH_SEQS = 4

D_BENCH_PATH = (PALAASH_ROOT / "stitching_large_to_small" / "results" / "llama"
               / "hard_factual" / "benches" / "bench_i08_j14_dev_exit_distill_add_unfroze.json")
OUT_CSV = HERE / "results_control_ABCD.csv"
CAPTURE_CACHE = HERE / "capture_control_i08_j14.npz"


def fit_items(n_bank: int, n_corpus: int) -> list[dict]:
    bank = [dict(p, _source="bank") for p in split_prompts(BANK, "fit")[:n_bank]]
    corpus = [dict(c, _source="corpus") for c in load_corpus("generic", n_corpus, seed=0)]
    return bank + corpus


@torch.no_grad()
def capture(lm_small, lm_large, items: list[dict]) -> dict:
    """Captures everything both B and C need: the teacher-forced sequence's
    3B top-k logits (target for both), plus the layer-14 (3B) / layer-8 (1B)
    residuals C's adapter maps between (B never uses X/Y at all, but capturing
    them costs nothing extra -- same forward pass)."""
    xs, ys, is_answer, prompt_index, position, tvals, tidx = [], [], [], [], [], [], []
    for pi, p in enumerate(items):
        system, framing = variant_for(pi, BANK.system)
        question = apply_framing(framing, p["question"])
        budget = BANK.teacher_answer_tokens if p["_source"] == "bank" else 48
        ids, n_prompt = teacher_sequence(lm_large, question, budget, system)
        sx, logits = layer_states(lm_large, ids, [J_LARGE], want_logits=True)
        sy, _ = layer_states(lm_small, ids, [I_SMALL], want_logits=False)
        n = ids.shape[1]
        xs.append(sx[J_LARGE])
        ys.append(sy[I_SMALL])
        v, k = topk_teacher(logits, DISTILL_TOPK)
        tvals.append(v)
        tidx.append(k)
        is_answer.extend([0] * n_prompt + [1] * (n - n_prompt))
        prompt_index.extend([pi] * n)
        position.extend(range(n))
        if pi % 100 == 0 or pi == len(items) - 1:
            print(f"  [capture] {pi + 1}/{len(items)}", flush=True)
    return {
        "X": np.concatenate(xs).astype(np.float32),
        "Y": np.concatenate(ys).astype(np.float32),
        "pid": np.array(prompt_index, np.int32),
        "is_answer": np.array(is_answer, np.int8),
        "position": np.array(position, np.int32),
        "t_vals": np.concatenate(tvals),
        "t_idx": np.concatenate(tidx),
    }


def save_capture(cap: dict, n_items: int) -> None:
    np.savez(CAPTURE_CACHE, X=cap["X"].astype(np.float16), Y=cap["Y"].astype(np.float16),
             pid=cap["pid"], is_answer=cap["is_answer"], position=cap["position"],
             t_vals=cap["t_vals"], t_idx=cap["t_idx"], n_items=n_items)


def load_capture_cache(n_items: int) -> dict | None:
    if not CAPTURE_CACHE.exists():
        return None
    z = np.load(CAPTURE_CACHE)
    if int(z["n_items"]) != n_items:
        return None
    return {"X": z["X"].astype(np.float32), "Y": z["Y"].astype(np.float32),
            "pid": z["pid"], "is_answer": z["is_answer"], "position": z["position"],
            "t_vals": z["t_vals"], "t_idx": z["t_idx"]}


def _sequences(pid, is_answer, position):
    out = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(position[idx], kind="stable")]
        if (is_answer[idx] == 1).any():
            out.append(idx)
    return out


def _exit_masks(idx, is_answer, preserve_prefix=PRESERVE_PREFIX):
    """Same as distill.py's `_masks` for train_mode="exit": every position
    except the preserved prefix is where B's fine-tuned tail runs (there is
    no "injection" for B -- it's the tail's own normal forward -- but the
    LOSS positions must match, so B and C are trained on the identical
    objective). Loss = answer positions + the last prompt position (the one
    that emits the first answer token)."""
    ans = is_answer[idx] == 1
    loss = ans.copy()
    first = int(np.argmax(ans)) if ans.any() else 0
    if first > 0:
        loss[first - 1] = True
    return loss


def train_tail_only(cap: dict, lm_small, i: int, epochs=EPOCHS, lr=TAIL_LR,
                    batch_seqs=BATCH_SEQS, seed: int = 0, val_frac: float = 0.2,
                    max_grad_norm: float = 1.0) -> dict:
    """Condition B: fine-tune 1B blocks i..end on KL-to-3B-teacher, with NO
    stitching -- the model runs its own normal forward end to end, reading
    its own embedding and its own blocks 0..i-1. Isolates whether D's gain
    needs the 3B's actual mid-layer representations, or is just fine-tuning."""
    device = lm_small.device
    stack = Stack(lm_small)
    Y, pid, is_answer, position = cap["Y"], cap["pid"], cap["is_answer"], cap["position"]
    t_vals, t_idx = cap["t_vals"], cap["t_idx"]
    # Y here is 1B's OWN layer-i residual under teacher forcing on the 3B's
    # answer text -- i.e. exactly what the model's own blocks 0..i-1 would
    # produce. Re-deriving it from scratch per step would need the small
    # model's embed + blocks 0..i-1 every time; using the captured value is
    # equivalent (blocks 0..i-1 are frozen and unmodified) and much cheaper.

    tail_params = []
    for layer in stack.layers[i:]:
        for p in layer.parameters():
            p.requires_grad_(True)
            tail_params.append(p)
    unexpected = [p for p in lm_small.model.parameters()
                 if p.requires_grad and id(p) not in {id(q) for q in tail_params}]
    if unexpected:
        raise SystemExit(f"{len(unexpected)} 1B parameters outside the intended tail have "
                         f"requires_grad=True -- freeze the model before calling this.")
    print(f"  [freeze check] tail params={sum(p.numel() for p in tail_params):,} "
          f"(blocks {i}..{stack.n_layers - 1})")

    opt = torch.optim.AdamW(tail_params, lr=lr)
    seqs = _sequences(pid, is_answer, position)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_frac)))
    val_ids = set(order[:n_val].tolist())
    train_seqs = [s for k, s in enumerate(seqs) if k not in val_ids]
    val_seqs = [s for k, s in enumerate(seqs) if k in val_ids]

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)

    def seq_loss(idx):
        loss_at = _exit_masks(idx, is_answer)
        h = to(Y[idx]).to(stack.dtype).unsqueeze(0)     # the model's OWN residual, untouched
        pos = to(position[idx]).long().unsqueeze(0)
        out = stack.run(h, pos, i, stack.n_layers, cache=None)
        logits = stack.lm_head(stack.base.norm(out))[0]
        sel = logits[to(np.flatnonzero(loss_at)).long()].float()
        return kl_to_teacher(sel, to(t_vals[idx][loss_at]), to(t_idx[idx][loss_at]))

    @torch.no_grad()
    def evaluate(which):
        return sum(float(seq_loss(idx)) for idx in which) / max(1, len(which))

    best = start_val = evaluate(val_seqs)
    best_state = [p.detach().clone() for p in tail_params]
    best_ep = 0
    print(f"  [train-B] {len(train_seqs)} train / {len(val_seqs)} val sequences, "
          f"unmodified-1B val loss {start_val:.4f}")

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(train_seqs))
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, si in enumerate(perm, 1):
            loss = seq_loss(train_seqs[si]) / batch_seqs
            loss.backward()
            tot += float(loss.detach()) * batch_seqs
            nb += 1
            if k % batch_seqs == 0 or k == len(perm):
                torch.nn.utils.clip_grad_norm_(tail_params, max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if k % 200 == 0:
                print(f"  [train-B]   epoch {ep} {k}/{len(perm)}  "
                      f"running train {tot / max(1, nb):.4f}", flush=True)
        v = evaluate(val_seqs)
        print(f"  [train-B] epoch {ep}: train {tot / max(1, nb):.4f}  val {v:.4f}  "
              f"(best {min(best, v):.4f})", flush=True)
        if v < best:
            best, best_ep = v, ep
            best_state = [p.detach().clone() for p in tail_params]

    with torch.no_grad():
        for p, saved in zip(tail_params, best_state):
            p.copy_(saved)
    for p in tail_params:
        p.requires_grad_(False)
    return {"val_loss_start": start_val, "val_loss_best": best, "best_epoch": best_ep}


@torch.no_grad()
def run_eval(runner, tokenizer, stop_ids, prompts: list[dict]) -> dict:
    rows = {}
    total_ms = 0.0
    for q in prompts:
        ids = build_prompt_ids(tokenizer, q["question"], runner.device, BANK.system)
        out = greedy(runner, ids, BANK.max_new_tokens, stop_ids)
        text = tokenizer.decode(out["token_ids"], skip_special_tokens=True).strip()
        ok = bool(score(text, q))
        rows[q["id"]] = ok
        total_ms += out["total_ms"]
    n = len(prompts)
    acc = sum(rows.values()) / n
    return {"accuracy": acc, "mean_total_ms": total_ms / n, "rows": rows}


def divergent_ids(a_rows: dict, l_rows: dict) -> list[str]:
    return sorted(pid for pid in a_rows if not a_rows[pid] and l_rows.get(pid))


def subset_acc(rows: dict, ids: list[str]) -> float:
    if not ids:
        return float("nan")
    return sum(rows[i] for i in ids) / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    n_bank = 5 if a.smoke else len(split_prompts(BANK, "fit"))
    n_corpus = 5 if a.smoke else N_CORPUS
    epochs = 1 if a.smoke else EPOCHS

    device = pick_device()
    print(f"[main] device={device}  i={I_SMALL}  j={J_LARGE}  mode={MODE}  "
          f"injection={INJECTION_MODE}  smoke={a.smoke}")
    lm_small = load_lm(SMALL_ID, "1B", device)
    lm_large = load_lm(LARGE_ID, "3B", device)
    stop_small, stop_large = eos_ids(lm_small), eos_ids(lm_large)

    items = fit_items(n_bank, n_corpus)
    print(f"[data] {len(items)} fit items ({n_bank} bank + {n_corpus} generic corpus, "
          f"matching D's exact recipe)")

    cap = None if a.smoke else load_capture_cache(len(items))
    if cap is not None:
        print(f"[capture] reusing cached capture ({len(cap['is_answer'])} rows)")
    else:
        print("[capture] teacher-forcing + tapping layer 14 (3B) / layer 8 (1B) ...")
        t0 = time.time()
        cap = capture(lm_small, lm_large, items)
        print(f"[capture] {len(cap['is_answer'])} rows in {time.time() - t0:.0f}s")
        if not a.smoke:
            save_capture(cap, len(items))
            print("[capture] cached")

    dev_prompts = split_prompts(BANK, "dev")
    print(f"[eval] dev split: {len(dev_prompts)} prompts")

    # ---- Condition A: 1B alone -------------------------------------------------
    print("[A] 1B alone")
    a_res = run_eval(FullRunner(lm_small, "1B"), lm_small.tokenizer, stop_small, dev_prompts)
    print("[ref] 3B alone")
    l_res = run_eval(FullRunner(lm_large, "3B"), lm_large.tokenizer, stop_large, dev_prompts)
    div_ids = divergent_ids(a_res["rows"], l_res["rows"])
    print(f"  1B: acc={a_res['accuracy']:.1%}   3B: acc={l_res['accuracy']:.1%}   "
          f"divergent (1B wrong, 3B right): {len(div_ids)}")

    print("[freeze] freezing all of 1B and 3B")
    for p in lm_small.model.parameters():
        p.requires_grad_(False)
    for p in lm_large.model.parameters():
        p.requires_grad_(False)

    # ---- Condition C: frozen stitch, exit+add -----------------------------------
    print(f"\n[C] frozen stitch (exit+add), i={I_SMALL} j={J_LARGE}")
    base = zero_map(PAIR.dim_large, PAIR.dim_small)
    arrays_c, meta_c, _ = distill_train(
        PAIR, BANK, I_SMALL, J_LARGE, base, lm_small,
        cap["X"], cap["Y"], cap["pid"], cap["is_answer"], cap["position"],
        cap["t_vals"], cap["t_idx"], lm_large=lm_large, epochs=epochs, lr=LR,
        batch_seqs=BATCH_SEQS, train_mode=TRAIN_MODE, injection_mode=INJECTION_MODE,
        unfreeze_tail=False, verbose=True)
    print(f"  [C] val KL: {meta_c['distill_val_loss_ridge']:.4f} -> "
          f"{meta_c['distill_val_loss_best']:.4f} @ epoch {meta_c['distill_best_epoch']}")
    adapter_c = TorchAdapter(arrays_c, {"small_layer_i": I_SMALL, "large_layer_j": J_LARGE,
                                        "injection_mode": INJECTION_MODE}, device)
    runner_c = LargeToSmallRunner(lm_small, lm_large, adapter_c, mode=MODE,
                                  injection_mode=INJECTION_MODE)
    c_res = run_eval(runner_c, lm_small.tokenizer, stop_small, dev_prompts)
    print(f"  [C] acc={c_res['accuracy']:.1%}  divergent_acc={subset_acc(c_res['rows'], div_ids):.1%}")

    # ---- Condition B: tail fine-tuned on 3B teacher, NO stitch ------------------
    print(f"\n[B] tail fine-tuned on 3B teacher, NO stitch, blocks {I_SMALL}..15")
    meta_b = train_tail_only(cap, lm_small, I_SMALL, epochs=epochs)
    print(f"  [B] val KL: {meta_b['val_loss_start']:.4f} -> "
          f"{meta_b['val_loss_best']:.4f} @ epoch {meta_b['best_epoch']}")
    b_res = run_eval(FullRunner(lm_small, "1B-tail-tuned"), lm_small.tokenizer, stop_small, dev_prompts)
    print(f"  [B] acc={b_res['accuracy']:.1%}  divergent_acc={subset_acc(b_res['rows'], div_ids):.1%}")

    # ---- Condition D: read from disk, never regenerated -------------------------
    d = None
    if D_BENCH_PATH.exists():
        with open(D_BENCH_PATH) as f:
            d = json.load(f)
        print(f"\n[D] read from disk: {D_BENCH_PATH.name}")
        print(f"  [D] acc={d['accuracy_stitch']:.1%}  small={d['accuracy_small']:.1%}  "
              f"divergent_acc={d['divergent']['accuracy_stitch']:.1%}  "
              f"beats_small_ci={d['beats_small_ci']}")
    else:
        print(f"\n[D] {D_BENCH_PATH} not found -- skipping (report A/B/C only)")

    rows = [
        {"label": "A_1B_alone", "accuracy": a_res["accuracy"],
        "divergent_accuracy": subset_acc(a_res["rows"], div_ids), "n_divergent": len(div_ids),
        "mean_ms_per_answer": a_res["mean_total_ms"], "notes": "baseline"},
        {"label": "B_tail_tuned_no_stitch", "accuracy": b_res["accuracy"],
        "divergent_accuracy": subset_acc(b_res["rows"], div_ids), "n_divergent": len(div_ids),
        "mean_ms_per_answer": b_res["mean_total_ms"],
        "notes": f"val_kl {meta_b['val_loss_start']:.3f}->{meta_b['val_loss_best']:.3f}"},
        {"label": "C_frozen_stitch_exit_add", "accuracy": c_res["accuracy"],
        "divergent_accuracy": subset_acc(c_res["rows"], div_ids), "n_divergent": len(div_ids),
        "mean_ms_per_answer": c_res["mean_total_ms"],
        "notes": f"val_kl {meta_c['distill_val_loss_ridge']:.3f}->{meta_c['distill_val_loss_best']:.3f}"},
    ]
    if d is not None:
        rows.append({"label": "D_unfrozen_stitch_exit_add_ONDISK",
                    "accuracy": d["accuracy_stitch"],
                    "divergent_accuracy": d["divergent"]["accuracy_stitch"],
                    "n_divergent": d["divergent"]["n"],
                    "mean_ms_per_answer": "", "notes": "read from existing sibling-package bench file, "
                                                       "not regenerated in this session"})
        rows.append({"label": "3B_alone_ref", "accuracy": l_res["accuracy"],
                    "divergent_accuracy": 1.0 if div_ids else float("nan"),
                    "n_divergent": len(div_ids), "mean_ms_per_answer": l_res["mean_total_ms"],
                    "notes": "reference ceiling"})

    if not a.smoke:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[done] wrote {OUT_CSV}")

    print("\n" + "=" * 70)
    print(f"{'condition':<30}{'accuracy':>10}{'divergent acc':>16}")
    for r in rows:
        da = r["divergent_accuracy"]
        print(f"{r['label']:<30}{r['accuracy']:>9.1%} {(f'{da:.1%}' if da==da else 'n/a'):>15}")
    print(f"\nn_divergent (1B wrong, 3B right, this session's own A/3B run): {len(div_ids)}")
    if d is not None:
        print("\nInterpretation:")
        print(f"  C vs A:        {'C >> A -> real frozen-stitch win' if c_res['accuracy'] - a_res['accuracy'] > 0.05 else 'C does not clearly beat A -- frozen stitch not shown to win here'}")
        print(f"  D vs A, D vs B: {'D >> A but D ~= B -> fine-tuning won, not stitching' if abs(d['accuracy_stitch'] - b_res['accuracy']) < 0.03 else 'D >> B -> unfreezing AND the 3B mid-layers both add something'}")


if __name__ == "__main__":
    main()
