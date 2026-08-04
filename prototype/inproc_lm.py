"""In-process, weight-owning LM for DSPy (proof-of-concept).

`InProcessLM` is a `dspy.BaseLM` subclass on the typed contract
(`forward_contract = "typed_lm"`). Unlike every other DSPy LM, it owns model
weights directly: the transformer lives in this Python process, on CPU, in RAM.
`forward()` runs `transformers` `generate()` locally -- no server, no GPU, no
network at inference.

Serialization is weight-owning: `dump_state()` emits the actual model WEIGHTS
as safetensors bytes (not a model id, not a path) plus the config and tokenizer
needed to rebuild, so the state is a fully self-contained artifact that
round-trips to an identical model in a fresh process.

Default model: PleIAs/Baguettotron (~321M params, Llama architecture,
ChatML-style chat template). CPU / float32 only in this PoC; a `device` slot is
threaded through so a later revision could select cuda/mps.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMTextPart, LMUsage

DEFAULT_MODEL = "PleIAs/Baguettotron"
# Baguettotron's ChatML template ends every assistant turn with <|im_end|>.
# The tokenizer defines no eos_token, so we stop generation on this id.
_IM_END_TOKEN = "<|im_end|>"


def _resolve_device(device: str) -> str:
    """Resolve the requested device slot to a concrete torch device string.

    The PoC is CPU-only; "auto" therefore resolves to "cpu". The cuda/mps
    branches exist so a later revision can flip on accelerators without
    changing the serialization or forward shape.
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _messages_to_chat(messages: list) -> list[dict[str, str]]:
    """Flatten typed LMMessages into the {role, content} dicts the chat template wants."""
    chat: list[dict[str, str]] = []
    for message in messages:
        text = message.text
        if text is None:
            # Concatenate any text parts; non-text parts are unsupported here.
            text = "".join(
                part.text for part in message.parts if getattr(part, "type", None) == "text"
            )
        chat.append({"role": message.role, "content": text})
    return chat


class InProcessLM(BaseLM):
    """A DSPy LM that owns Baguettotron's weights in-process on CPU.

    This is a weight-owning LM: the model parameters live in this object and
    travel inside `dump_state()` as safetensors bytes. There is no server and no
    network call at inference time.
    """

    forward_contract = "typed_lm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        _model: Any | None = None,
        _tokenizer: Any | None = None,
        _config: Any | None = None,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = 256,
        cache: bool = False,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        """Build an in-process LM.

        Args:
            model: HF model id to load when weights are not injected directly.
            device: Device slot. "cpu" (default) or "auto"; cuda/mps are
                resolvable but untested in this CPU-only PoC.
            _model / _tokenizer / _config: Pre-built objects, used by
                `load_state` to rebuild from serialized weights without touching
                the network. Not part of the public API.
        """
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=num_retries,
            **kwargs,
        )
        self.device = _resolve_device(device)
        self._device_slot = device  # preserve the requested slot for round-trip

        if _model is not None and _tokenizer is not None:
            self.hf_config = _config if _config is not None else _model.config
            self.tokenizer = _tokenizer
            self.hf_model = _model
        else:
            # Load once, in-process. local_files_only is off only for the very
            # first fetch; after dump_state the artifact is fully self-contained.
            self.hf_config = AutoConfig.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)

        self.hf_model.to(self.device)
        self.hf_model.eval()

        end_id = self.tokenizer.convert_tokens_to_ids(_IM_END_TOKEN)
        self._eos_token_id = end_id if isinstance(end_id, int) and end_id >= 0 else None

    # -- capabilities: honest flags for a tiny local base model ----------------

    @property
    def supports_function_calling(self) -> bool:
        # No native tool-calling; adapters must prompt for tools.
        return False

    @property
    def supports_reasoning(self) -> bool:
        # The chat template emits a <think> block, but there is no native
        # reasoning-token API surfaced through DSPy, so report False.
        return False

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        # Buffered generate only; DSPy replays the finished response as events.
        return False

    @property
    def supported_params(self) -> set[str]:
        return {"temperature", "max_tokens"}

    # -- forward: local greedy/temp generation ---------------------------------

    def _build_inputs(self, request: LMRequest):
        chat = _messages_to_chat(request.messages)
        encoded = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers may return a bare tensor or a BatchEncoding dict.
        if hasattr(encoded, "shape"):
            input_ids = encoded
        else:
            input_ids = encoded["input_ids"]
        return input_ids.to(self.device)

    def forward(self, request: LMRequest) -> LMResponse:
        input_ids = self._build_inputs(request)
        prompt_len = input_ids.shape[1]

        max_new = request.config.max_tokens or self.kwargs.get("max_tokens") or 256
        temperature = request.config.temperature
        if temperature is None:
            temperature = self.kwargs.get("temperature")

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new),
            "pad_token_id": self._eos_token_id,
        }
        if self._eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self._eos_token_id

        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False)  # greedy / temperature 0

        with torch.no_grad():
            generated = self.hf_model.generate(input_ids, **gen_kwargs)

        new_tokens = generated[0, prompt_len:]
        finish_reason = "stop"
        if self._eos_token_id is not None and len(new_tokens) and new_tokens[-1].item() == self._eos_token_id:
            new_tokens = new_tokens[:-1]
        elif len(new_tokens) >= int(max_new):
            finish_reason = "length"

        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        usage = LMUsage(
            input_tokens=int(prompt_len),
            output_tokens=int(len(new_tokens)),
        )
        return LMResponse(
            model=self.model,
            outputs=[
                LMOutput(
                    parts=[LMTextPart(text=text)],
                    finish_reason=finish_reason,
                    truncated=finish_reason == "length",
                )
            ],
            usage=usage,
        )

    # -- weight-owning serialization -------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        """Serialize the full model as a self-contained artifact.

        The weights travel as safetensors bytes (base64-encoded so the state is
        JSON-serializable, matching how DSPy persists LM state). Also carried:
        the HF config JSON and the tokenizer files, so a fresh process rebuilds
        the exact same model with no network access. No api_key, no path, no
        model-id-only shortcut.
        """
        # 1. Weights -> safetensors bytes. Contiguous CPU tensors only.
        # Baguettotron ties lm_head.weight to embed_tokens.weight; safetensors
        # rejects shared storage, so we drop the tied duplicate here and let the
        # model re-tie it on load (tie_weights runs in from_config).
        state_dict = self.hf_model.state_dict()
        tied_keys = set(getattr(self.hf_model, "_tied_weights_keys", None) or [])
        tensors = {
            name: param.detach().cpu().contiguous()
            for name, param in state_dict.items()
            if name not in tied_keys
        }
        weights_bytes = safetensors_save(tensors)

        # 2. Config as JSON.
        config_json = self.hf_config.to_json_string()

        # 3. Tokenizer files, saved to a temp dir then read back as bytes.
        with tempfile.TemporaryDirectory() as tmp:
            self.tokenizer.save_pretrained(tmp)
            tokenizer_files = {
                path.name: base64.b64encode(path.read_bytes()).decode("ascii")
                for path in sorted(Path(tmp).iterdir())
                if path.is_file()
            }

        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "inproc_lm_version": 1,
            "model": self.model,
            "model_type": self.model_type,
            "device": self._device_slot,
            "cache": self.cache,
            "num_retries": self.num_retries,
            "temperature": self.kwargs.get("temperature"),
            "max_tokens": self.kwargs.get("max_tokens"),
            "weights_format": "safetensors",
            "weights_b64": base64.b64encode(weights_bytes).decode("ascii"),
            "config_json": config_json,
            "tokenizer_files_b64": tokenizer_files,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> "InProcessLM":
        """Rebuild an InProcessLM from `dump_state` output, no network needed."""
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        state.pop("inproc_lm_version", None)
        state.pop("weights_format", None)

        # Rebuild config.
        config = AutoConfig.from_pretrained(
            _write_json_to_tempfile(state.pop("config_json"))
        )

        # Rebuild tokenizer from its serialized files.
        tokenizer_files = state.pop("tokenizer_files_b64")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name, b64 in tokenizer_files.items():
                (tmp_path / name).write_bytes(base64.b64decode(b64))
            tokenizer = AutoTokenizer.from_pretrained(tmp)

        # Rebuild model from config, then load weights from safetensors bytes.
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(torch.float32)
        weights_bytes = base64.b64decode(state.pop("weights_b64"))
        tensors = safetensors_load(weights_bytes)
        missing, unexpected = model.load_state_dict(tensors, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected weight keys on load: {unexpected[:5]}")
        # Re-tie shared weights (e.g. lm_head <- embed_tokens) that we dropped on
        # save; any `missing` keys should be exactly those tied duplicates.
        model.tie_weights()

        return cls(
            model=state.get("model", DEFAULT_MODEL),
            device=state.get("device", "cpu"),
            _model=model,
            _tokenizer=tokenizer,
            _config=config,
            model_type=state.get("model_type", "chat"),
            temperature=state.get("temperature"),
            max_tokens=state.get("max_tokens", 256),
            cache=state.get("cache", False),
            num_retries=state.get("num_retries", 0),
        )


def _write_json_to_tempfile(config_json: str) -> str:
    """AutoConfig.from_pretrained wants a directory containing config.json."""
    tmp = tempfile.mkdtemp()
    (Path(tmp) / "config.json").write_text(config_json)
    return tmp
