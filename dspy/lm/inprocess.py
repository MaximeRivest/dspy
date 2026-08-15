"""The in-process engine: weights loaded in this process, no server.

`InProcessEngine` is a structural lm15 engine — `complete(Request) ->
Response` plus a replayed `stream()` — whose backend is a `transformers`
model living in the program's own process. No base class, no server, no
credential: the model string (an HF repo id or a local directory) is all
it needs. `from_pretrained` downloads to the HF cache on first use,
exactly the Ollama/LM Studio convenience, and the device resolves to
`cuda` > `mps` > `cpu` unless pinned.

The engine also declares `programir_weight_spec()` — the structural
weight-baking protocol — so a program bound to an in-process LM bakes
its weights (safetensors + rebuild config + tokenizer + declared ties)
into the ProgramIR artifact with no extra code. Weight ties are read
from the loaded model's shared storage and DECLARED into `tying.json`;
the artifact never asks a loader to rediscover them.

Loading is lazy: constructing the engine (and the `dspy.LM` over it)
costs nothing; the first call pays for the load. Text-only chat is the
contract — image or tool parts refuse loudly rather than degrade.
"""

from __future__ import annotations

from typing import Any

from lm15 import Message, Request, Response, Usage, response_to_events

from dspy.core.errors import LMError

__all__ = ["InProcessEngine"]

#: Completion-token cap applied when a request does not carry one.
DEFAULT_MAX_TOKENS = 512


class InProcessEngine:
    """A transformers-backed lm15 engine running in this process.

    Args:
        model: An HF repo id (`"PleIAs/Baguettotron"`) or a local model
            directory. Repo ids download to the HF cache on first use.
        device: `"cuda"`, `"mps"`, or `"cpu"`. `None` picks the best
            available, in that order.
        max_tokens: Completion-token cap used when a request carries none.

    Examples:
        ```python
        lm = dspy.LM("hf:PleIAs/Baguettotron")           # builds this engine
        lm = dspy.LM("/models/baguettotron", device="cpu")
        ```
    """

    # deps: torch, transformers, safetensors
    forward_contract = "typed_lm"

    def __init__(self, model: str, *, device: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model_id = model
        self.requested_device = device
        self.default_max_tokens = max_tokens
        self.transformer: Any = None
        self.tokenizer: Any = None
        self.device: str | None = None

    # ---- the lm15 engine contract -------------------------------------

    def complete(self, request: Request) -> Response:
        """Run one chat completion on the loaded weights."""
        self._ensure_loaded()
        import torch

        messages = self._chat_messages(request)
        input_ids = self._encode(messages)
        config = request.config
        max_new = config.max_tokens or self.default_max_tokens

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if config.temperature and config.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                generate_kwargs["top_p"] = config.top_p
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.transformer.generate(input_ids, **generate_kwargs)
        new_tokens = output[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        finish_reason = "length" if len(new_tokens) >= max_new else "stop"
        text, stopped = _truncate_at_stop(text, config.stop)
        if stopped:
            finish_reason = "stop"

        return Response(
            id=None,
            model=self.model_id,
            message=Message.assistant(text),
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=int(input_ids.shape[1]),
                output_tokens=len(new_tokens),
                total_tokens=int(input_ids.shape[1]) + len(new_tokens),
            ),
        )

    def stream(self, request: Request) -> Any:
        """Replay the finished response as canonical stream events."""
        return response_to_events(self.complete(request))

    # ---- the structural weight-baking protocol ------------------------

    def programir_weight_spec(self) -> dict[str, Any]:
        """Declare the baked-weights slot for the ProgramIR exporter.

        Loads the model if the first bake arrives before the first call.
        Ties are read from the live model's shared storage and declared —
        the artifact carries them as data (`tying.json`), it never asks
        a loader to rediscover them.
        """
        self._ensure_loaded()
        return {
            "model_attribute": "transformer",
            "tokenizer_attribute": "tokenizer",
            "weights_identity": self.model_id,
            "engine": "transformers",
            "device": self.device or "cpu",
            "frozen": False,
            "weight_ref": "base",
            "ties": self._shared_ties(),
        }

    # ---- internals ----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self.transformer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise LMError(
                f"In-process LM {self.model_id!r} needs the 'torch' and 'transformers' "
                "packages. Install them, or bind a served model instead."
            ) from error

        device = self.requested_device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.transformer = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.transformer.to(device)
        self.transformer.eval()
        self.device = device

    def _chat_messages(self, request: Request) -> list[dict[str, str]]:
        """Flatten the request to role/content dicts; refuse non-text parts."""
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            text = message.text
            if text is None:
                kinds = ", ".join(type(part).__name__ for part in message.parts)
                raise LMError(
                    f"In-process LM {self.model_id!r} speaks text-only chat; a "
                    f"{message.role!r} message carries non-text parts ({kinds}). "
                    "Use a served model for tool calls or media."
                )
            messages.append({"role": message.role, "content": text})
        return messages

    def _encode(self, messages: list[dict[str, str]]) -> Any:
        """Encode via the chat template when the tokenizer has one."""
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # A base model with no template: plain concatenation, no
            # invented markers.
            prompt = "\n\n".join(m["content"] for m in messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.to(self.transformer.device)

    def _shared_ties(self) -> list[dict[str, str]]:
        """Declare tensors sharing storage: first name (sorted) is the source."""
        state = self.transformer.state_dict()
        groups: dict[Any, list[str]] = {}
        for name in sorted(state):
            tensor = state[name]
            key = tensor.data_ptr() if hasattr(tensor, "data_ptr") else id(tensor)
            groups.setdefault(key, []).append(name)
        return [
            {"target": target, "source": names[0]}
            for names in groups.values()
            if len(names) > 1
            for target in names[1:]
        ]

    def __repr__(self) -> str:
        loaded = "loaded" if self.transformer is not None else "not loaded"
        return f"InProcessEngine(model={self.model_id!r}, device={self.device or self.requested_device!r}, {loaded})"


def _truncate_at_stop(text: str, stop: Any) -> tuple[str, bool]:
    """Cut the text at the earliest stop string; report whether one hit."""
    if not stop:
        return text, False
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = min((text.find(s) for s in stops if s and text.find(s) != -1), default=-1)
    if cut == -1:
        return text, False
    return text[:cut], True
