"""Large->small stitched decoding, and the checks that gate reporting.

    prompt -> large embed + large blocks 0..j-1 -> adapter -> small blocks
    i..end -> small norm -> small lm_head -> token

Small blocks 0..i-1 and large blocks j..end never run during decode. Both models
keep a KV cache over the layers they do run.

The sliced layer loop (`Stack`), the KV-cached baselines, and the greedy decode
come from `common.decoding`, which both stitching packages share; only the
direction-specific runner and checks live here.

The attention sink now sits on the *small* stream, because that is the stream
being overwritten. Position 0 carries a residual far larger than any other and
the adapter cannot reproduce it, so the path re-derives it by running the
skipped small blocks over that one token — exact, because attention is causal.

Checks
------
The accuracy numbers are worthless if the injection is plumbed wrong, and a
wrong injection usually still produces fluent text, so it does not announce
itself. Four checks gate every report:

  full_stack_*        the hand-rolled loop reproduces HF's own forward
  identity_injection  feeding the small model its OWN layer-i residual at block
                      i reproduces the unmodified small model exactly. This is
                      the one that pins the convention: it fails if `i` is off
                      by one, if the residual is taken after the block instead
                      of before it, or if the resumed slice is wrong.
  offbyone_control    feeding layer i+1's residual at block i must DIFFER.
                      Without it, `identity_injection` could pass on a path that
                      silently ignores the injected value altogether.
  adapter_reload      the saved coefficients round-trip and score finitely.
"""

from __future__ import annotations

import torch

from common.decoding import (
    FullRunner, Stack, check_baseline_matches_hf, check_full_stack, greedy,
)
from common.model_utils import LM, build_prompt_ids
from stitching_large_to_small.adapter import TorchAdapter
from stitching_large_to_small.config import INJECTION_MODE, PRESERVE_PREFIX


class LargeToSmallRunner:
    """Large prefix -> adapter -> small suffix, KV-cached on both sides.

    `exit`  the prompt goes through the stitch too: the small model's blocks
            0..i-1 run over the sink position only, and every small KV the
            suffix reads is the adapter's reconstruction of the large model's
            reading of the prompt. This is the mode that actually transfers the
            large model's prompt comprehension.
    `warm`  the small model prefills the prompt itself, so its prompt KVs are
            its own and the adapter supplies the residual only at generated
            positions. Cheaper to get right, but the large model's contribution
            is limited to the tokens being generated.

    Neither mode is fast. Both run j large blocks and (n_small - i) small blocks
    per token, plus a full large-model prefix prefill. Accuracy is the point;
    `evaluate` reports the cost so it cannot be quietly dropped.
    """

    MODES = ("exit", "warm")

    def __init__(self, lm_small: LM, lm_large: LM, adapter: TorchAdapter,
                 preserve_prefix: int = PRESERVE_PREFIX, mode: str = "exit",
                 injection_mode: str | None = None):
        if mode not in self.MODES:
            raise SystemExit(f"mode must be one of {self.MODES}")
        self.small, self.large = Stack(lm_small), Stack(lm_large)
        self.adapter = adapter
        self.i, self.j = adapter.i, adapter.j
        self.preserve_prefix = preserve_prefix
        self.mode = mode
        # Read off the adapter's own sidecar by default, so a bench never has
        # to be told out-of-band which formula the map it just loaded expects.
        self.injection_mode = injection_mode or adapter.meta.get(
            "injection_mode", INJECTION_MODE)
        self.label = (f"stitch-{mode}-largeL{self.j}->smallL{self.i}"
                      + ("-add" if self.injection_mode == "add" else ""))
        self.device = lm_small.device
        self.small_cache = self.large_cache = None

    def large_features(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Run large blocks 0..j-1 and return the residual at the input to block
        j — the same quantity `hidden_states[j]` names, which is what capture
        recorded and the adapter was fit on."""
        return self.large.run(self.large.embed(ids), pos, 0, self.j, self.large_cache)

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        n_prompt = ids.shape[1]
        pos = torch.arange(n_prompt, device=ids.device)[None]

        # The large prefix cache is needed in both modes: it supplies layer j at
        # every position the stitch decodes.
        self.large_cache = self.large.new_cache()
        xl = self.large_features(ids, pos)
        self.small_cache = self.small.new_cache()

        if self.mode == "warm":
            h = self.small.run(self.small.embed(ids), pos, 0, self.small.n_layers,
                               self.small_cache)
            return self.small.head(h)

        if self.injection_mode == "add":
            # `add` needs the small model's own reading of the whole prompt to
            # correct, not just of the preserved sink — so blocks 0..i-1 run
            # over every prompt position here (populating `small_cache` for
            # those layers too, which `step` extends token by token). This is
            # what removes the `warm` first-token ceiling under `exit`: the
            # correction is now present at the last prompt position, which is
            # the one that decides the first generated token.
            own = self.small.run(self.small.embed(ids), pos, 0, self.i,
                                 self.small_cache)
            correction = self.adapter(xl).to(self.small.dtype)
            n = min(self.preserve_prefix, n_prompt)
            if n > 0:   # the sink stays exactly `own`, uncorrected — same
                correction[:, :n] = 0.0   # invariant `replace` mode's splice keeps
            h = own + correction
        else:
            h = self.adapter(xl).to(self.small.dtype)
            n = min(self.preserve_prefix, n_prompt)
            if n > 0:   # the sink, re-derived exactly from the small model's own blocks
                h[:, :n] = self.small.run(self.small.embed(ids[:, :n]), pos[:, :n],
                                          0, self.i, cache=None)
        h = self.small.run(h, pos, self.i, self.small.n_layers, self.small_cache)
        return self.small.head(h)

    def step(self, token_id: int, position: int) -> torch.Tensor:
        ids = torch.tensor([[token_id]], device=self.device)
        pos = torch.tensor([[position]], device=self.device)
        xl = self.large_features(ids, pos)
        if self.injection_mode == "add":
            # Extends the blocks-0..i-1 cache `prefill` started (exit) or the
            # full-stack cache `prefill` already filled (warm) by one position,
            # exactly the way the small model's own decode would.
            own = self.small.run(self.small.embed(ids), pos, 0, self.i,
                                 self.small_cache)
            h = own + self.adapter(xl).to(self.small.dtype)
        else:
            h = self.adapter(xl).to(self.small.dtype)
        h = self.small.run(h, pos, self.i, self.small.n_layers, self.small_cache)
        return self.small.head(h)

    def active_params(self) -> int:
        """Large prefix (no unembedding — its logits are never formed) + adapter
        + small suffix, per decoded token. `add` also runs small blocks 0..i-1
        every step to produce the residual being corrected, which `replace`
        does not — leaving it out would understate `add`'s real cost."""
        extra = self.small.n_params(0, self.i, head=False) if self.injection_mode == "add" else 0
        return (self.large.n_params(0, self.j, head=False)
                + self.adapter.n_params
                + self.small.n_params(self.i, self.small.n_layers)
                + extra)


# ── direction-specific checks ─────────────────────────────────────────────────
@torch.no_grad()
def _resume_small(lm_small: LM, ids: torch.Tensor, inject: torch.Tensor,
                  i: int) -> torch.Tensor:
    """Run the small model from block i with `inject` as the residual, and
    return the next-token logits. This is exactly the stitched suffix, with the
    adapter's output replaced by whatever the caller supplies."""
    stack = Stack(lm_small)
    pos = torch.arange(ids.shape[1], device=ids.device)[None]
    h = stack.run(inject, pos, i, stack.n_layers, cache=None)
    return stack.head(h)[0].float()


@torch.no_grad()
def check_identity_injection(lm_small: LM, ids: torch.Tensor, i: int) -> dict:
    """Feed the small model its OWN layer-i residual at block i.

    Nothing has been changed, so the logits must match the unmodified model.
    This is what pins down the injection convention — that `hidden_states[i]` is
    the *input* to block i, and that resuming at `i` means running blocks
    i..end. An off-by-one anywhere in that chain breaks it.
    """
    ref = lm_small.model(ids).logits[0, -1].float()
    own = lm_small.model(ids, output_hidden_states=True).hidden_states[i]
    got = _resume_small(lm_small, ids, own, i)
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    return {"check": "identity_injection", "layer_i": i, "rel_l2": rel,
            "max_abs_logit_diff": float((ref - got).abs().max()),
            "argmax_match": int(ref.argmax()) == int(got.argmax()),
            "passed": rel < 5e-3 and int(ref.argmax()) == int(got.argmax())}


@torch.no_grad()
def check_offbyone_control(lm_small: LM, ids: torch.Tensor, i: int) -> dict:
    """Negative control: inject layer i+1's residual at block i.

    That is the wrong tensor, so the result must *differ* from the unmodified
    model. If it does not, the resumed slice is ignoring what it was handed and
    `identity_injection` proved nothing.
    """
    ref = lm_small.model(ids).logits[0, -1].float()
    wrong = lm_small.model(ids, output_hidden_states=True).hidden_states[i + 1]
    got = _resume_small(lm_small, ids, wrong, i)
    rel = float((ref - got).norm() / (ref.norm() + 1e-12))
    return {"check": "offbyone_control", "layer_injected": i + 1, "at_block": i,
            "rel_l2": rel, "argmax_match": int(ref.argmax()) == int(got.argmax()),
            # A real difference, not a rounding one. The identity check's
            # tolerance is 5e-3, so anything at least ten times that is
            # unambiguously a different computation.
            "passed": rel > 5e-2}


@torch.no_grad()
def check_prefix_exact(lm_small: LM, ids: torch.Tensor, i: int, n: int) -> dict:
    """The sink splice claims a prefix-only run through small blocks 0..i-1
    equals the small model's own layer-i residual at those positions. Causality
    makes that an identity in exact arithmetic; in bf16 the pass criterion is
    bit-identity against HF's own prefix-only forward, with HF's identical shape
    drift recorded alongside to show whose it is."""
    stack = Stack(lm_small)
    hs = lambda x: lm_small.model(x, output_hidden_states=True).hidden_states[i].float()
    full, hf_pre = hs(ids)[:, :n], hs(ids[:, :n])
    pos = torch.arange(n, device=ids.device)[None]
    ours = stack.run(stack.embed(ids[:, :n]), pos, 0, i, cache=None).float()
    rel = lambda a, b: float((a - b).norm() / (b.norm() + 1e-12))
    return {"check": "prefix_exact", "n_positions": n,
            "rel_l2_vs_hf_prefix": rel(ours, hf_pre),
            "shape_drift_rel_l2": rel(ours, full),
            "hf_own_shape_drift_rel_l2": rel(hf_pre, full),
            "passed": bool((ours == hf_pre).all())}


@torch.no_grad()
def check_zero_add_reproduces_small(lm_small: LM, lm_large: LM, i: int, j: int,
                                    mode: str, n_tokens: int = 8,
                                    probe: str = "What is the capital city of New Zealand?") -> dict:
    """The `injection_mode="add"` analogue of `identity_injection`, checked end
    to end through the runner rather than through one hand-built resume.

    A zero adapter (W=0, b=0) adds nothing, so `own_i + 0` must be exactly the
    small model's own residual at every injected position, which must decode
    to exactly the small model's own greedy generation. This is what licenses
    reading `injection_mode="add"` epoch 0 as "the unmodified small model" —
    the claim the zero-map warm start in `adapter.fit` depends on.
    """
    from stitching_large_to_small.adapter import TorchAdapter, zero_map
    dim_small = lm_small.model.config.hidden_size
    dim_large = lm_large.model.config.hidden_size
    adapter = TorchAdapter(zero_map(dim_large, dim_small),
                           {"small_layer_i": i, "large_layer_j": j}, lm_small.device)
    runner = LargeToSmallRunner(lm_small, lm_large, adapter, mode=mode,
                                injection_mode="add")
    ids = build_prompt_ids(lm_small.tokenizer, probe, lm_small.device)
    ref = greedy(FullRunner(lm_small, "small"), ids, n_tokens, None)["token_ids"]
    got = greedy(runner, ids, n_tokens, None)["token_ids"]
    return {"check": "zero_add_reproduces_small", "mode": mode,
            "ref_tokens": ref, "got_tokens": got, "passed": ref == got}


def run_checks(lm_small: LM, lm_large: LM, i: int, j: int,
               probe: str = "What is the capital city of New Zealand?",
               injection_mode: str = "replace", modes: tuple[str, ...] = ()) -> list[dict]:
    ids = build_prompt_ids(lm_small.tokenizer, probe, lm_small.device)
    checks = [
        check_full_stack(lm_small, ids, "small"),
        check_full_stack(lm_large, ids, "large"),
        check_baseline_matches_hf(lm_small, probe, "small"),
        check_identity_injection(lm_small, ids, i),
        check_offbyone_control(lm_small, ids, i),
        check_prefix_exact(lm_small, ids, i, max(1, PRESERVE_PREFIX)),
    ]
    if injection_mode == "add":
        checks += [check_zero_add_reproduces_small(lm_small, lm_large, i, j, m)
                  for m in (modes or ("exit", "warm"))]
    return checks
