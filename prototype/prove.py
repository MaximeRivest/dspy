"""Proof harness for InProcessLM.

Prints EXPECTED vs ACTUAL for each claim and exits nonzero on any failure.

Claims proven:
  (a) build + one greedy forward on a fixed prompt -> capture generated token ids
  (b) dump_state -> bytes -> reload into a FRESH InProcessLM -> IDENTICAL output
  (c) trainability: one (prompt, target) pair, single grad step, loss finite AND
      at least one weight tensor changed
  (d) no network at inference; no api_key in serialized state
"""

from __future__ import annotations

import copy
import json
import socket
import sys

import torch

import dspy
from prototype.inproc_lm import InProcessLM

FAILURES: list[str] = []


def check(name: str, expected, actual) -> None:
    ok = expected == actual
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"       EXPECTED: {expected}")
    print(f"       ACTUAL:   {actual}")
    if not ok:
        FAILURES.append(name)


def check_true(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAILURES.append(name)


FIXED_PROMPT = "What is the capital of France? Answer in one word."


def _greedy_ids(lm: InProcessLM) -> list[int]:
    request = dspy.LMRequest(
        model=lm.model,
        messages=[dspy.User(FIXED_PROMPT)],
        config=dspy.LMConfig(temperature=0.0, max_tokens=16),
    )
    input_ids = lm._build_inputs(request)
    with torch.no_grad():
        generated = lm.hf_model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=lm._eos_token_id,
            eos_token_id=lm._eos_token_id,
        )
    return generated[0, input_ids.shape[1]:].tolist()


def main() -> int:
    print("=" * 70)
    print("(a) BUILD + ONE GREEDY FORWARD")
    print("=" * 70)
    lm = InProcessLM(device="cpu")
    response = lm(
        dspy.LMRequest(
            model=lm.model,
            messages=[dspy.User(FIXED_PROMPT)],
            config=dspy.LMConfig(temperature=0.0, max_tokens=16),
        )
    )
    check_true("forward returns LMResponse", isinstance(response, dspy.LMResponse))
    check_true("response has >=1 output", len(response.outputs) >= 1)
    check_true("output has text", bool(response.text is not None))
    print(f"       GENERATED TEXT: {response.text!r}")
    ids_before = _greedy_ids(lm)
    print(f"       GREEDY TOKEN IDS: {ids_before}")

    print("=" * 70)
    print("(b) DUMP -> BYTES -> RELOAD FRESH -> IDENTICAL OUTPUT")
    print("=" * 70)
    state = lm.dump_state()
    # Prove it survives an actual JSON serialization round-trip (bytes over wire).
    blob = json.dumps(state).encode("utf-8")
    size_mb = len(blob) / (1024 * 1024)
    print(f"       SERIALIZED ARTIFACT SIZE: {size_mb:.2f} MB")
    restored_state = json.loads(blob.decode("utf-8"))
    fresh = InProcessLM.load_state(restored_state)
    check_true("fresh is InProcessLM", isinstance(fresh, InProcessLM))
    ids_after = _greedy_ids(fresh)
    check("greedy token ids identical after reload", ids_before, ids_after)

    print("=" * 70)
    print("(d) NO api_key IN STATE  +  NO NETWORK AT INFERENCE")
    print("=" * 70)
    flat_keys = set(state.keys())
    check_true("no 'api_key' key in serialized state", "api_key" not in flat_keys)
    check_true(
        "no credential-like keys in state",
        not any("key" in k.lower() and k != "tokenizer_files_b64" and "weights" not in k.lower() for k in flat_keys),
        detail=f"keys={sorted(flat_keys)}",
    )

    # Block all network, then run a forward on the FRESH (reloaded) model.
    _orig_socket = socket.socket

    def _blocked(*a, **k):
        raise AssertionError("network access attempted during inference")

    socket.socket = _blocked  # type: ignore[assignment]
    try:
        offline_resp = fresh(
            dspy.LMRequest(
                model=fresh.model,
                messages=[dspy.User(FIXED_PROMPT)],
                config=dspy.LMConfig(temperature=0.0, max_tokens=8),
            )
        )
        check_true("inference works with sockets blocked", offline_resp.text is not None)
    except AssertionError as exc:
        check_true("inference works with sockets blocked", False, detail=str(exc))
    finally:
        socket.socket = _orig_socket  # type: ignore[assignment]

    print("=" * 70)
    print("(c) TRAINABILITY: ONE GRAD STEP MUTATES WEIGHTS")
    print("=" * 70)
    train_lm = InProcessLM(device="cpu")
    tok = train_lm.tokenizer
    model = train_lm.hf_model
    model.train()

    prompt_text = "The capital of France is"
    target_text = " Paris."
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    target_ids = tok(target_text, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # cross-entropy only on the target continuation

    # Snapshot a weight tensor before the step.
    watched_name = next(n for n, p in model.named_parameters() if p.requires_grad)
    before = model.state_dict()[watched_name].detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.zero_grad()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    optimizer.step()

    after = model.state_dict()[watched_name].detach().clone()
    loss_val = loss.item()
    changed = not torch.equal(before, after)
    max_delta = (after - before).abs().max().item()

    check_true("loss is finite", bool(torch.isfinite(loss).item()), detail=f"loss={loss_val:.4f}")
    check_true("watched weight tensor changed", changed, detail=f"tensor={watched_name} max|delta|={max_delta:.3e}")

    print("=" * 70)
    if FAILURES:
        print(f"RESULT: FAILED ({len(FAILURES)} check(s)): {FAILURES}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
