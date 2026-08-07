"""Author the ProgramIR contract corpus through the public DSPy exporter.

This module is an on-server build input, not a pytest module. Run it from a
DSPy checkout whose Hugging Face cache contains PleIAs/Baguettotron.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import dspy
from dspy.clients.openai_compat_lm import _OpenAICompatLM


class CorpusInProcessLM(dspy.BaseLM):
    """Own the Baguettotron model and declare its ProgramIR weight layout."""

    forward_contract = "typed_lm"

    def __init__(self, model_id="PleIAs/Baguettotron", device="cpu"):
        # deps: torch, transformers, safetensors
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model=model_id, cache=False)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            local_files_only=True,
        ).to(device)
        self.hf_model.eval()

    def programir_weight_spec(self):
        raw = getattr(self.hf_model, "_tied_weights_keys", None) or []
        targets = list(raw.keys()) if isinstance(raw, dict) else list(raw)
        parameters = dict(self.hf_model.named_parameters(remove_duplicate=False))
        ties = []
        for target in targets:
            pointer = parameters[target].data_ptr()
            source = next(
                name
                for name, parameter in parameters.items()
                if name != target and parameter.data_ptr() == pointer
            )
            ties.append({"target": target, "source": source})
        return {
            "model_attribute": "hf_model",
            "tokenizer_attribute": "tokenizer",
            "weights_identity": self.model,
            "engine": "transformers",
            "device": self.device,
            "frozen": False,
            "weight_ref": "base",
            "ties": ties,
        }

    def forward(self, request):
        import torch

        messages = [{"role": message.role, "content": message.text or ""} for message in request.messages]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.hf_model.generate(inputs, max_new_tokens=request.config.max_tokens or 64)
        text = self.tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=True)
        return dspy.LMResponse.from_text(text, model=self.model)


class SharedTasks(dspy.Module):
    def __init__(self):
        self.sentiment = dspy.Predict(
            dspy.Signature("text -> sentiment").with_instructions(
                "Classify the sentiment of the text as positive or negative."
            ),
            temperature=0.0,
            max_tokens=8,
        )
        self.language = dspy.Predict(
            dspy.Signature("text -> lang").with_instructions(
                "Identify the language of the text as english or french."
            ),
            temperature=0.0,
            max_tokens=8,
        )

    def forward(self, text):
        sentiment = self.sentiment(text=text)
        language = self.language(text=text)
        return sentiment


class Classifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(
            dspy.Signature("question -> category").with_instructions(
                "Classify as creative or factual."
            ),
            temperature=0.0,
            max_tokens=300,
        )

    def forward(self, question):
        category = self.classify(question=question)
        return category


class Drafter(dspy.Module):
    def __init__(self):
        self.classifier = Classifier()
        self.draft_factual = dspy.Predict(
            dspy.Signature("question -> draft_answer").with_instructions(
                "Answer the factual question accurately."
            ),
            temperature=0.0,
            max_tokens=300,
        )
        self.draft_creative = dspy.Predict(
            dspy.Signature("question -> draft_answer").with_instructions(
                "Answer with one imaginative sentence."
            ),
            temperature=0.0,
            max_tokens=300,
        )

    def forward(self, question):
        cat = self.classifier(question=question)
        if cat.category == "creative":
            draft = self.draft_creative(question=question)
        else:
            draft = self.draft_factual(question=question)
        return draft


class NestedAnswerer(dspy.Module):
    def __init__(self):
        self.drafter = Drafter()
        self.polish = dspy.Predict(
            dspy.Signature("question, draft_answer -> answer").with_instructions(
                "Rewrite the draft into a polished answer."
            ),
            temperature=0.0,
            max_tokens=300,
        )

    def forward(self, question):
        draft = self.drafter(question=question)
        final = self.polish(question=question, draft_answer=draft.draft_answer)
        return final


def calculator(arg: str) -> str:
    """Evaluate the integer multiplication used by the corpus prompt."""
    left, right = arg.split("*")
    return str(int(left) * int(right))


def lookup(arg: str) -> str:
    """Return one capital from a self-contained lookup table."""
    capitals = {"france": "Paris", "canada": "Ottawa"}
    return capitals.get(arg.lower(), "unknown")


class MiniReAct(dspy.Module):
    def __init__(self):
        self.react = dspy.Predict(
            "question, trajectory -> thought, tool_name, tool_arg",
            temperature=0.0,
            max_tokens=300,
        )
        self.extract = dspy.Predict("question, trajectory -> answer", temperature=0.0, max_tokens=300)
        self.tools = {"calculator": calculator, "lookup": lookup}

    def forward(self, question):
        if question == "":
            raise ToolError("empty question")
        trajectory = "(empty)"
        for step in range(3):
            pred = self.react(question=question, trajectory=trajectory)
            if pred.tool_name == "finish":
                break
            try:
                observation = self.tools[pred.tool_name](arg=pred.tool_arg)
            except ToolError:
                observation = "tool failed or unknown; use another tool or finish"
            trajectory = (
                trajectory
                + "\nthought: "
                + pred.thought
                + "\naction: "
                + pred.tool_name
                + "["
                + pred.tool_arg
                + "]"
                + "\nobservation: "
                + observation
            )
        final = self.extract(question=question, trajectory=trajectory)
        return final


class InProcessInterpreter:
    def programir_profile(self):
        return {
            "language": "python",
            "runtime": {"identity": "cpython", "version": "3.11"},
            "contract": "execute(code)->result",
            "namespace_policy": "fresh-empty-no-builtins",
            "result_convention": "result variable",
            "vars_marshaling": "json-values",
            "packages": [],
            "resource_limits": {"max_attempts": 2},
            "isolation_floor": "none",
            "placement": {
                "rung": "in_process",
                "contract": "execute(code)->result",
                "endpoint_ref": None,
                "isolation": "none",
                "credential_ref": None,
            },
        }

    def __call__(self, code):
        namespace = {"__builtins__": {}}
        exec(code, namespace)
        return str(namespace["result"])


class MiniRLM(dspy.Module):
    def __init__(self):
        self.gen = dspy.Predict("question -> code", temperature=0.0, max_tokens=24)
        self.answer = dspy.Predict("question, result -> answer", temperature=0.0, max_tokens=24)
        self.interpreter = InProcessInterpreter()

    def forward(self, question):
        result = ""
        attempts = 0
        while result == "":
            if attempts == 2:
                break
            pred = self.gen(question=question)
            try:
                result = self.interpreter(code=pred.code)
            except InterpreterError:
                result = ""
            attempts = attempts + 1
        final = self.answer(question=question, result=result)
        return final


def exact_match(example, prediction, field) -> float:
    """Score one prediction by exact field equality."""
    return float(example[field] == prediction[field])


def _terse_template(name="terse"):
    return dspy.TemplateAdapter(
        messages=[
            {
                "role": "system",
                "content": (
                    "{instruction} Given {% for f in inputs separator=', ' %}{f.name}{% endfor %}, "
                    "reply with one line `{% for f in outputs separator=' ' %}{f.name}: <{f.name}>"
                    "{% endfor %}` and nothing else."
                ),
            },
            {
                "role": "user",
                "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}",
            },
            {
                "role": "assistant",
                "content": "</think>\n{% for f in outputs %}{f.name}:{% endfor %}",
            },
        ],
        parse_mode="full_text",
        name=name,
    )


def _remote_lm(*, local: bool):
    return _OpenAICompatLM(
        model="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" if local else "cerebras/llama3.1-8b",
        base_url="http://localhost:8000/v1" if local else "https://api.cerebras.ai/v1",
        api_key="corpus-build-placeholder",
        require_auth=True,
        cache=False,
    )


def _predict(adapter, lm):
    program = dspy.Predict("question -> answer", temperature=0.0, max_tokens=64)
    program.set_adapter(adapter)
    program.set_lm(lm)
    return program


def _export(program, destination, *, metric=None, devset=None):
    if destination.exists():
        shutil.rmtree(destination)
    dspy.export(program, destination, metric=metric, devset=devset)
    print(destination, flush=True)
    gc.collect()


def build_all(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    baked_lm = CorpusInProcessLM()

    _export(_predict(dspy.ChatAdapter(), baked_lm), root / "01")
    _export(_predict(dspy.XMLAdapter(), baked_lm), root / "02")
    _export(_predict(dspy.JSONAdapter(), baked_lm), root / "03")
    _export(_predict(dspy.XMLAdapter(), _remote_lm(local=True)), root / "09")
    _export(_predict(dspy.XMLAdapter(), _remote_lm(local=False)), root / "10")

    shared = SharedTasks()
    shared.set_lm(baked_lm)
    shared.set_adapter(_terse_template())
    _export(shared, root / "12", metric=exact_match, devset=[])

    nested = NestedAnswerer()
    nested.set_lm(_remote_lm(local=True))
    nested.set_adapter(dspy.XMLAdapter())
    _export(nested, root / "13")

    react = MiniReAct()
    react.set_lm(_remote_lm(local=True))
    react.set_adapter(dspy.XMLAdapter())
    _export(react, root / "14")

    rlm = MiniRLM()
    rlm.set_lm(baked_lm)
    rlm.set_adapter(_terse_template("rlm-terse"))
    _export(rlm, root / "15")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    build_all(parser.parse_args().output)
