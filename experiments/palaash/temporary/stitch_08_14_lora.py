"""Stitch small-model layer 8 to large-model layer 14, with the three fixes
proposed for why plain task-loss distillation capped out at the 1B baseline:

1. MORE, MORE DIVERSE FIT DATA. Previously: 211 hard_factual questions, 817
   answer rows. Here: those 211 PLUS N_ALPACA real, diverse Stanford Alpaca
   instructions (instruction-only items, no gold answer needed -- the 3B
   model's own greedy continuation is the target, exactly as for the bank
   items). Not the 10k-50k of full literature-scale runs (that would not
   finish in this environment), but a real order-of-magnitude increase using
   genuinely diverse text, not templated repeats.
2. NORMALISED ADAPTER. `NormedAdapter` applies RMSNorm(x) * gamma before the
   linear map, addressing activation-scale drift between the 3B layer-14 and
   1B layer-8 residual streams that a bare `x @ W + b` does not correct for.
   W is zero-initialised, so -- like every adapter in this folder's history
   -- epoch 0 is exactly the unmodified 1B model.
3. LoRA ON THE 1B TOP. Blocks 8..15 of the 1B model get rank-8 LoRA adapters
   on every attention and MLP projection (q/k/v/o, gate/up/down), trained
   jointly with the bridge adapter. B is zero-initialised (standard LoRA
   practice), so this also starts as a no-op. This directly targets the
   "frozen 1B attractor" ceiling: the earlier plain-distill result could only
   ever match 1B, not exceed it, because the frozen suffix had no way to
   learn how to use richer 3B-derived signal it wasn't trained to expect.

Everything else matches the rest of this folder's history: TASK LOSS ONLY
(KL to the 3B teacher through the suffix), `injection_mode="add"`, scored on
the same disjoint EVAL_QUESTIONS. No ridge, no Procrustes, anywhere.

Point (4) from the critique this responds to -- that 0.97x-1.15x speeds
(rather than 1.5-2x) are a local MPS/Python dispatch-overhead artifact, not a
methodology bug -- is not something this script can fix: it would need a
different runtime (batched CUDA kernels), not a different adapter.

Fits on FIT_QUESTIONS + an Alpaca subset, scores 1B alone / 3B alone / the
stitched hybrid on EVAL_QUESTIONS. One (i, j) cell. Writes
results_08_14_lora.csv. Caches the Alpaca subset to alpaca_subset.json after
first download so re-runs do not re-fetch it.

    python stitch_08_14_lora.py               # full run (~1400 fit items)
    python stitch_08_14_lora.py --smoke        # tiny smoke test first
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PALAASH_ROOT = HERE.parent
sys.path.insert(0, str(PALAASH_ROOT))

from common.decoding import FullRunner, Stack, eos_ids, greedy  # noqa: E402
from common.model_utils import build_prompt_ids, load_lm, pick_device  # noqa: E402
from common.scoring import score  # noqa: E402
from common.templates import apply_framing, variant_for  # noqa: E402
from stitching_large_to_small.config import (  # noqa: E402
    BANKS, DISTILL_BATCH_SEQS, DISTILL_MAX_GRAD_NORM, DISTILL_TOPK,
    DISTILL_VAL_FRAC, PAIRS, PRESERVE_PREFIX, SEED,
)
from stitching_large_to_small.data import (  # noqa: E402
    layer_states, teacher_sequence, topk_teacher,
)
from stitching_large_to_small.distill import kl_to_teacher  # noqa: E402
from stitching_large_to_small.stitch import LargeToSmallRunner  # noqa: E402

from questions import EVAL_QUESTIONS, FIT_QUESTIONS  # noqa: E402

PAIR = PAIRS["llama"]
SMALL_ID, LARGE_ID = PAIR.small_id, PAIR.large_id
I_SMALL, J_LARGE = 8, 14
BANK = BANKS["hard_factual"]
INJECTION_MODE = "add"

N_ALPACA = 1200
CORPUS_ANSWER_TOKENS = 48
ALPACA_CACHE = HERE / "alpaca_subset.json"
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

LORA_RANK = 8
LORA_ALPHA = 16.0
LORA_LR = 1e-4
ADAPTER_LR = 2e-6      # same order as the sibling package's DISTILL_LR
EPOCHS = 4
OUT_CSV = HERE / "results_08_14_lora.csv"
CAPTURE_CACHE = HERE / "capture_08_14.npz"

LORA_PROJ_PATHS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def load_alpaca(n: int, seed: int = 0) -> list[str]:
    if ALPACA_CACHE.exists():
        with open(ALPACA_CACHE) as f:
            cached = json.load(f)
        if len(cached) >= n:
            return cached[:n]
    with urllib.request.urlopen(ALPACA_URL, timeout=30) as r:
        data = json.loads(r.read())
    no_input = [d["instruction"] for d in data if not d.get("input")]
    rng = np.random.default_rng(seed)
    rng.shuffle(no_input)
    subset = no_input[:n]
    with open(ALPACA_CACHE, "w") as f:
        json.dump(subset, f)
    return subset


class NormedAdapter(torch.nn.Module):
    """RMSNorm(x) * gamma @ W + b. W zero-initialised so, under
    injection_mode="add", the untrained adapter contributes exactly zero --
    the same "epoch 0 == unmodified small model" property every adapter in
    this folder's history has had, just with a normalisation step that a
    bare affine map does not."""

    def __init__(self, dim_in: int, dim_out: int, device: str):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim_in, device=device, dtype=torch.float32))
        self.W = torch.nn.Parameter(torch.zeros(dim_in, dim_out, device=device, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(dim_out, device=device, dtype=torch.float32))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)
        xn = (x / rms) * self.gamma
        return xn @ self.W + self.b


class LoRALinear(torch.nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update:
    y = base(x) + scale * x @ A^T @ B^T. B is zero-initialised, so this is a
    no-op until trained (standard LoRA practice)."""

    def __init__(self, base: torch.nn.Linear, rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        device = base.weight.device
        self.A = torch.nn.Parameter(
            torch.randn(rank, base.in_features, device=device, dtype=torch.float32) * 0.01)
        self.B = torch.nn.Parameter(
            torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x.to(torch.float32) @ self.A.T @ self.B.T) * self.scale
        return base_out + delta.to(base_out.dtype)


def install_lora(lm_small, start_i: int, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> list:
    """Wrap every attention/MLP projection in blocks start_i..end with LoRA.
    Must be called AFTER freezing lm_small.model's parameters wholesale, so
    the only trainable pieces left are the freshly created A, B tensors."""
    layers = Stack(lm_small).layers
    params = []
    for layer in layers[start_i:]:
        for path in LORA_PROJ_PATHS:
            *parents, leaf = path.split(".")
            node = layer
            for p in parents:
                node = getattr(node, p)
            base = getattr(node, leaf)
            assert isinstance(base, torch.nn.Linear), f"{path} is not nn.Linear"
            wrapped = LoRALinear(base, rank, alpha)
            setattr(node, leaf, wrapped)
            params += [wrapped.A, wrapped.B]
    return params


@torch.no_grad()
def capture_fit(lm_small, lm_large, items: list[tuple[str, int]]) -> dict:
    """items: (question, answer_token_budget) pairs, in the fixed order
    template variety is applied over (matches stitching_large_to_small/data.py's
    convention of varying framing across bank AND corpus items alike)."""
    xs, ys, is_answer, prompt_index, position, tvals, tidx = [], [], [], [], [], [], []
    for pi, (question_raw, answer_tokens) in enumerate(items):
        system, framing = variant_for(pi, BANK.system)
        question = apply_framing(framing, question_raw)
        ids, n_prompt = teacher_sequence(lm_large, question, answer_tokens, system)
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


def load_capture(n_items: int) -> dict | None:
    """Returns the cached capture only if it was made from the same number of
    fit items -- a different N_ALPACA/n_bank means different rows entirely."""
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


def train_lora_normed(cap: dict, lm_small, adapter: NormedAdapter, lora_params: list,
                      epochs: int = EPOCHS, adapter_lr: float = ADAPTER_LR,
                      lora_lr: float = LORA_LR, batch_seqs: int = DISTILL_BATCH_SEQS,
                      seed: int = SEED, val_frac: float = DISTILL_VAL_FRAC,
                      max_grad_norm: float = DISTILL_MAX_GRAD_NORM) -> dict:
    device = lm_small.device
    stack = Stack(lm_small)
    X, Y = cap["X"], cap["Y"]
    pid, is_answer, position = cap["pid"], cap["is_answer"], cap["position"]
    t_vals, t_idx = cap["t_vals"], cap["t_idx"]

    trainable_ids = {id(p) for p in adapter.parameters()} | {id(p) for p in lora_params}
    unexpected = [p for p in lm_small.model.parameters()
                 if p.requires_grad and id(p) not in trainable_ids]
    if unexpected:
        raise SystemExit(f"{len(unexpected)} 1B parameters have requires_grad=True outside "
                         f"the adapter+LoRA set -- freeze the model before install_lora().")
    print(f"  [freeze check] adapter params={sum(p.numel() for p in adapter.parameters()):,}  "
          f"LoRA params={sum(p.numel() for p in lora_params):,}  "
          f"1B total params={sum(p.numel() for p in lm_small.model.parameters()):,}")

    opt = torch.optim.AdamW([
        {"params": list(adapter.parameters()), "lr": adapter_lr},
        {"params": lora_params, "lr": lora_lr},
    ])

    seqs = _sequences(pid, is_answer, position)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(seqs))
    n_val = max(1, int(round(len(seqs) * val_frac)))
    val_ids = set(order[:n_val].tolist())
    train_seqs = [s for k, s in enumerate(seqs) if k not in val_ids]
    val_seqs = [s for k, s in enumerate(seqs) if k in val_ids]

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)

    def seq_loss(idx):
        ans = is_answer[idx] == 1              # train_mode="warm": write+loss both = answer rows
        h = to(Y[idx]).to(torch.float32).unsqueeze(0).clone()
        inj = to(np.flatnonzero(ans)).long()
        correction = adapter(to(X[idx][ans]).to(torch.float32))
        h[0, inj] = h[0, inj] + correction      # injection_mode="add"
        pos = to(position[idx]).long().unsqueeze(0)
        out = stack.run(h.to(stack.dtype), pos, I_SMALL, stack.n_layers, cache=None)
        logits = stack.lm_head(stack.base.norm(out))[0]
        sel = logits[inj].float()
        return kl_to_teacher(sel, to(t_vals[idx][ans]), to(t_idx[idx][ans]))

    @torch.no_grad()
    def evaluate(which):
        return sum(float(seq_loss(idx)) for idx in which) / max(1, len(which))

    best = start_val = evaluate(val_seqs)
    best_ep = 0
    best_adapter_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    best_lora_state = [p.detach().clone() for p in lora_params]
    print(f"  [train] {len(train_seqs)} train / {len(val_seqs)} val sequences, "
          f"zero-start val loss {start_val:.4f}")

    clip_params = list(adapter.parameters()) + lora_params
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
                torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if k % 200 == 0:
                print(f"  [train]   epoch {ep} {k}/{len(perm)}  "
                      f"running train {tot / max(1, nb):.4f}", flush=True)
        v = evaluate(val_seqs)
        print(f"  [train] epoch {ep}: train {tot / max(1, nb):.4f}  val {v:.4f}  "
              f"(best {min(best, v):.4f})", flush=True)
        if v < best:
            best, best_ep = v, ep
            best_adapter_state = {k: p.detach().clone() for k, p in adapter.state_dict().items()}
            best_lora_state = [p.detach().clone() for p in lora_params]

    adapter.load_state_dict(best_adapter_state)
    with torch.no_grad():
        for p, saved in zip(lora_params, best_lora_state):
            p.copy_(saved)
    return {"val_loss_start": start_val, "val_loss_best": best, "best_epoch": best_ep}


@torch.no_grad()
def run_eval(runner, tokenizer, stop_ids) -> dict:
    n_ok, total_ms, total_tokens = 0, 0.0, 0
    for q in EVAL_QUESTIONS:
        ids = build_prompt_ids(tokenizer, q["question"], runner.device, BANK.system)
        out = greedy(runner, ids, BANK.max_new_tokens, stop_ids)
        text = tokenizer.decode(out["token_ids"], skip_special_tokens=True).strip()
        n_ok += score(text, q)
        total_ms += out["total_ms"]
        total_tokens += out["n_generated"]
    n = len(EVAL_QUESTIONS)
    return {"accuracy": n_ok / n, "mean_total_ms": total_ms / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (10 fit items, rank-4 LoRA, 1 epoch) to check correctness")
    a = ap.parse_args()

    n_alpaca = 5 if a.smoke else N_ALPACA
    n_bank = 5 if a.smoke else len(FIT_QUESTIONS)
    rank = 4 if a.smoke else LORA_RANK
    epochs = 1 if a.smoke else EPOCHS

    device = pick_device()
    print(f"[main] device={device}  i={I_SMALL}  j={J_LARGE}  smoke={a.smoke}  "
          f"bank_fit={n_bank}  alpaca={n_alpaca}  lora_rank={rank}  epochs={epochs}")
    lm_small = load_lm(SMALL_ID, "1B", device)
    lm_large = load_lm(LARGE_ID, "3B", device)
    stop_small, stop_large = eos_ids(lm_small), eos_ids(lm_large)

    print(f"[data] loading {n_alpaca} Alpaca instructions ...")
    alpaca = load_alpaca(n_alpaca)
    items = ([(q["question"], BANK.teacher_answer_tokens) for q in FIT_QUESTIONS[:n_bank]]
             + [(instr, CORPUS_ANSWER_TOKENS) for instr in alpaca])
    print(f"[data] {len(items)} total fit items ({n_bank} bank + {len(alpaca)} Alpaca)")

    cap = None if a.smoke else load_capture(len(items))
    if cap is not None:
        print(f"[capture] reusing cached capture at {CAPTURE_CACHE.name} "
              f"({len(cap['is_answer'])} rows)")
    else:
        print("[capture] fitting on bank + Alpaca items (template variety on) ...")
        t0 = time.time()
        cap = capture_fit(lm_small, lm_large, items)
        print(f"[capture] {len(cap['is_answer'])} rows ({int(cap['is_answer'].sum())} answer rows) "
              f"in {time.time() - t0:.0f}s")
        if not a.smoke:
            save_capture(cap, len(items))
            print(f"[capture] cached to {CAPTURE_CACHE.name}")

    print("[baseline] 1B alone")
    small_res = run_eval(FullRunner(lm_small, "1B"), lm_small.tokenizer, stop_small)
    print("[baseline] 3B alone")
    large_res = run_eval(FullRunner(lm_large, "3B"), lm_large.tokenizer, stop_large)
    print(f"  1B: acc={small_res['accuracy']:.1%}  ms/answer={small_res['mean_total_ms']:.0f}")
    print(f"  3B: acc={large_res['accuracy']:.1%}  ms/answer={large_res['mean_total_ms']:.0f}")

    print("[freeze] freezing all of 1B and 3B before installing adapter + LoRA")
    for p in lm_small.model.parameters():
        p.requires_grad_(False)
    for p in lm_large.model.parameters():
        p.requires_grad_(False)

    print(f"[lora] installing rank-{rank} LoRA on 1B blocks {I_SMALL}..{PAIR.n_layers_small - 1}")
    lora_params = install_lora(lm_small, I_SMALL, rank=rank)
    adapter = NormedAdapter(PAIR.dim_large, PAIR.dim_small, device)

    print("[train] task-loss only, adapter + LoRA jointly")
    t0 = time.time()
    meta = train_lora_normed(cap, lm_small, adapter, lora_params, epochs=epochs)
    print(f"[train] val KL: zero-start {meta['val_loss_start']:.4f} -> "
          f"best {meta['val_loss_best']:.4f} @ epoch {meta['best_epoch']}  "
          f"({time.time() - t0:.0f}s)")

    print("[eval] hybrid (normed adapter + LoRA)")
    adapter.i, adapter.j = I_SMALL, J_LARGE
    adapter.meta = {"injection_mode": INJECTION_MODE}
    runner = LargeToSmallRunner(lm_small, lm_large, adapter, mode="warm",
                                injection_mode=INJECTION_MODE)
    hybrid_res = run_eval(runner, lm_small.tokenizer, stop_small)

    rows = []
    for label, res, kl in (
        ("1B_alone", small_res, ""),
        ("3B_alone", large_res, ""),
        ("hybrid_lora_normed_i08_j14", hybrid_res, meta["val_loss_best"]),
    ):
        acc_pct = res["accuracy"] / large_res["accuracy"] * 100 if large_res["accuracy"] else float("nan")
        speed = large_res["mean_total_ms"] / res["mean_total_ms"]
        rows.append({"label": label, "n_eval": len(EVAL_QUESTIONS), "accuracy": res["accuracy"],
                    "accuracy_pct_of_3b": acc_pct, "mean_ms_per_answer": res["mean_total_ms"],
                    "speed_relative_to_3b": speed, "distill_val_kl": kl,
                    "n_fit_items": len(items)})
        print(f"  {label:<28} acc={res['accuracy']:.1%}  ({acc_pct:5.1f}% of 3B)  "
              f"speed={speed:.2f}x")

    if not a.smoke:
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[done] wrote {OUT_CSV}")
    else:
        print("\n[done] smoke test finished (no CSV written)")


if __name__ == "__main__":
    main()
