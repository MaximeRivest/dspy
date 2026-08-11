"""RLM: the mini recursive-language-model loop, authored plain.

The model writes Python; a DECLARED interpreter leaf runs it; the loop
repeats until a round produces a value (or the literal cap breaks it);
the extract predictor finishes. The corpus MiniRLM's architecture, made
a usable module and written as an ordinary readable forward (node-set
v0.4, the inputs-bag envelope, D-041):

- the forward is signature-polymorphic: `def forward(self, **inputs)`
  splats the module's own input record into the gen and extract
  predictors (`**inputs`, the v0.4 record envelope);
- the interpreter is a declared leaf (a D-033 structural profile in
  component 7); the default runtime is `InProcessInterpreter` — exec,
  empty builtins, `result`-variable convention;
- the loop is a While with a LITERAL cap: `max_iters` bakes in at compile
  time (named in `ir_literals` so edits recompile);
- a failed round is the typed `InterpreterError`, caught in-forward and
  turned into a transcript note the next round can read;
- the transcript is a plain string grown by concatenation — every round
  sees the code and output so far.

This is deliberately NOT the 817-line legacy RLM (sub-LM tools, batched
queries, sandbox marshaling); it is the smallest RLM that earns its
name. The forward below IS the artifact's forward, and the equivalence
test's control arm runs this exact Python.
"""

from __future__ import annotations

from dspy.core.errors import InterpreterError
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.programir.interpreters import InProcessInterpreter
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import ensure_signature, make_signature

__all__ = ["RLM"]

_RESERVED_FIELDS = ("transcript", "code", "result")


class RLM(Module):
    """Solve a task by writing and running Python, round by round.

    Args:
        signature: The task's signature (`"question -> answer"` or a
            `dspy.Signature`).
        max_iters: The round cap, baked into the compiled forward as a
            literal (named in `ir_literals` so edits recompile).
        interpreter: The interpreter runtime — any object with a D-033
            `programir_profile()` and a `__call__(code=...)` contract.
            Defaults to `InProcessInterpreter` (exec, empty builtins,
            `result`-variable convention).

    Examples:
        ```python
        rlm = dspy.RLM("question -> answer", max_iters=4)
        rlm.set_lm(dspy.LM("openai/gpt-4o-mini"))
        prediction = rlm(question="What is 6 * 7?")
        ```
    """

    ir_literals = ("max_iters",)

    def __init__(self, signature, max_iters: int = 8, interpreter=None):
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.interp = interpreter if interpreter is not None else InProcessInterpreter()

        collisions = sorted(set(signature.fields) & set(_RESERVED_FIELDS))
        if collisions:
            raise ValueError(
                f"RLM reserves the field name(s) {collisions} for its own loop channels; rename them "
                f"in your signature (reserved: {', '.join(_RESERVED_FIELDS)})"
            )

        self.gen = Predict(self._gen_signature())
        self.extract = Predict(self._extract_signature())

    def forward(self, **inputs):
        result = ""
        transcript = ""
        rounds = 0
        while result == "":
            if rounds == self.max_iters:
                break
            rounds = rounds + 1
            step = self.gen(**inputs, transcript=transcript)
            try:
                output = self.interp(code=step.code)
            except InterpreterError:  # the engine's typed error table
                output = ""
            note = transcript + "\n>>> " + step.code
            if output == "":
                transcript = note + "\n(error: the code failed; fix it and try again)"
            else:
                transcript = note + "\n" + output
            result = output
        final = self.extract(**inputs, transcript=transcript, result=result)
        return final

    def _gen_signature(self):
        signature = self.signature
        inputs = ", ".join(f"`{name}`" for name in signature.input_fields)
        outputs = ", ".join(f"`{name}`" for name in signature.output_fields)
        lines = []
        if signature.instructions:
            lines.append(signature.instructions)
            lines.append("")
        lines.extend(
            [
                f"You are working toward producing {outputs} from the input field(s) {inputs}, with a "
                "Python interpreter at your disposal.",
                "Each round, write one Python snippet in the `code` field; it runs immediately, and the "
                "transcript shows every previous snippet with its output.",
                "The interpreter has NO builtins and NO imports: plain expressions, arithmetic, string "
                "operations, loops.",
                "Assign your final value to a variable named `result`. A round whose code fails, or does "
                "not assign `result`, shows an error note in the transcript; fix it in the next round.",
                "The first round that produces a value ends the loop.",
            ]
        )
        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["transcript"] = (str, InputField(desc="every previous code snippet and its output"))
        fields["code"] = (str, OutputField(desc="the next Python snippet to run; assign `result` when done"))
        return make_signature(fields, "\n".join(lines), signature_name="RLMStep")

    def _extract_signature(self):
        signature = self.signature
        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["transcript"] = (str, InputField(desc="the completed run: code snippets and outputs"))
        fields["result"] = (str, InputField(desc="the value the final round computed (empty if none)"))
        for name, field in signature.output_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, OutputField(desc=desc))
        return make_signature(fields, signature.instructions, signature_name="RLMExtract")
