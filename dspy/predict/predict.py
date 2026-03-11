"""Prompt a language model with named inputs and get named outputs back.

`Predict` pairs a signature (which declares input and output fields)
with a language model.  `ChainOfThought`, `ReAct`, and other modules
build on top of `Predict`.
"""

import logging
import random

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)

UNSAFE_LM_STATE_KEYS = {"api_base", "base_url", "model_list"}


def _sanitize_lm_state(lm_state: dict, allow_unsafe_lm_state: bool) -> dict:
    """Strip sensitive keys from a serialized LM config unless explicitly allowed."""
    if allow_unsafe_lm_state:
        return lm_state

    unsafe_keys = sorted(UNSAFE_LM_STATE_KEYS.intersection(lm_state))

    if not unsafe_keys:
        return lm_state

    sanitized_lm_state = {k: v for k, v in lm_state.items() if k not in UNSAFE_LM_STATE_KEYS}
    logger.warning(
        "Ignoring unsafe LM config key(s) during state load: %s. "
        "Pass allow_unsafe_lm_state=True to preserve these keys for trusted files.",
        unsafe_keys,
    )
    return sanitized_lm_state


class Predict(Module, Parameter):
    """Prompt a language model with named inputs and get named outputs back.

    Supply a signature — either a string like `"question -> answer"`
    or a `dspy.Signature` class — then call the module with your
    inputs as keyword arguments.  Returns a `Prediction` whose
    attributes are the output fields declared in the signature.

    Args:
        signature: A string (e.g. `"question -> answer"`) or a
            `dspy.Signature` subclass that declares input and output
            fields.
        callbacks: Optional callback handlers for instrumentation.
        **config: LM parameters such as `temperature` or
            `max_tokens`, applied to every call.  Override for a
            single call by passing `config={...}` at call time.

    Returns:
        (Prediction): Attributes correspond to the signature's output
            fields.  For example, if the signature declares an
            `answer` output field, access it with `result.answer`.

    Examples:
        String signature:

        >>> import dspy
        >>> dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # doctest: +SKIP
        >>> predict = dspy.Predict("question -> answer")
        >>> result = predict(question="What is the capital of France?")  # doctest: +SKIP
        >>> result.answer  # doctest: +SKIP
        'Paris'

        Class-based signature (the docstring becomes the instruction):

        >>> class Translate(dspy.Signature):
        ...     '''Translate the text to the target language.'''
        ...     text: str = dspy.InputField()
        ...     language: str = dspy.InputField()
        ...     translation: str = dspy.OutputField()
        >>> translate = dspy.Predict(Translate)
        >>> translate(text="Hello", language="French")  # doctest: +SKIP
        Prediction(translation='Bonjour')

    See Also:
        [`dspy.ChainOfThought`][dspy.ChainOfThought]: Adds a reasoning
            step before the output.
        
        [`dspy.ReAct`][dspy.ReAct]: Interleaves reasoning with tool use.

        [`dspy.Signature`][dspy.Signature]: Declare input and output
            fields.
        
        [`dspy.LM`][dspy.LM]: Configure the language model.
    """

    def __init__(self, signature: str | type[Signature], callbacks: list[BaseCallback] | None = None, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.reset()

    def reset(self):
        """Clear demos, traces, train data, and the per-instance LM."""
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self, json_mode=True):
        """Serialize this module's learnable state to a dict.

        Captures demos, traces, train data, the signature state, and
        the LM configuration.  Restore later with `load_state`.

        Args:
            json_mode: If `True`, convert demo `Example` objects
                to plain dicts for JSON serialization.

        Returns:
            (dict): A JSON-serializable snapshot of the module.

        Examples:
            Save state, then restore it later (or on another machine):

            >>> import dspy
            >>> predict = dspy.Predict("question -> answer")
            >>> state = predict.dump_state()
            >>> predict.load_state(state)  # doctest: +ELLIPSIS
            Predict(...)

            See the [saving tutorial](https://dspy.ai/tutorials/saving/)
            for full save/load workflows.
        """
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                demo[field] = serialize_object(demo[field])

            if isinstance(demo, dict) or not json_mode:
                state["demos"].append(demo)
            else:
                state["demos"].append(demo.toDict())

        state["signature"] = self.signature.dump_state()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    def load_state(self, state: dict, *, allow_unsafe_lm_state: bool = False) -> "Predict":
        """Restore state from a dict produced by `dump_state`.

        By default, sensitive LM keys (`api_base`, `base_url`,
        `model_list`) are stripped for safety.  Pass
        `allow_unsafe_lm_state=True` only when loading trusted files.

        Args:
            state: Dict previously returned by `dump_state`.
            allow_unsafe_lm_state: If `True`, preserve sensitive LM
                config keys.  Only use with trusted data.

        Returns:
            (Predict): `self`, for method chaining.
        """
        excluded_keys = ["signature", "extended_signature", "lm"]
        for name, value in state.items():
            # `excluded_keys` are fields that go through special handling.
            if name not in excluded_keys:
                setattr(self, name, value)

        self.signature = self.signature.load_state(state["signature"])
        sanitized_lm_state = _sanitize_lm_state(state["lm"], allow_unsafe_lm_state) if state["lm"] else None
        self.lm = LM(**sanitized_lm_state) if sanitized_lm_state else None

        if "extended_signature" in state:  # legacy, up to and including 2.5, for CoT.
            raise NotImplementedError("Loading extended_signature is no longer supported in DSPy 2.6+")

        return self

    def _get_positional_args_error_message(self):
        input_fields = list(self.signature.input_fields.keys())
        return (
            "Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments "
            f"that match your signature input fields: '{', '.join(input_fields)}'. For example: "
            f"`predict({input_fields[0]}=input_value, ...)`."
        )

    def __call__(
        self,
        *args,
        config: dict | None = None,
        signature: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        """Call the predictor with keyword arguments matching the input fields.

        Pass one keyword argument for each input field in the signature.
        Use `config`, `signature`, `demos`, or `lm` to override this
        predictor for one call. Returns a `Prediction` whose
        attributes match the output fields.
        """
        if args:
            raise ValueError(self._get_positional_args_error_message())

        call_kwargs = dict(inputs)
        if config is not None:
            call_kwargs["config"] = config
        if signature is not None:
            call_kwargs["signature"] = signature
        if demos is not None:
            call_kwargs["demos"] = demos
        if lm is not None:
            call_kwargs["lm"] = lm

        return super().__call__(**call_kwargs)

    async def acall(
        self,
        *args,
        config: dict | None = None,
        signature: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        call_kwargs = dict(inputs)
        if config is not None:
            call_kwargs["config"] = config
        if signature is not None:
            call_kwargs["signature"] = signature
        if demos is not None:
            call_kwargs["demos"] = demos
        if lm is not None:
            call_kwargs["lm"] = lm

        return await super().acall(**call_kwargs)

    def _forward_preprocess(
        self,
        *,
        config: dict | None = None,
        signature: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        # Extract the privileged keyword arguments.
        assert "new_signature" not in inputs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(self.signature if signature is None else signature)
        demos = self.demos if demos is None else demos
        config = {**self.config, **({} if config is None else config)}

        # Get the right LM to use.
        lm = self.lm if lm is None else lm
        lm = lm or settings.lm

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            # Many users mistakenly use `dspy.configure(lm="openai/gpt-4o-mini")` instead of
            # `dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))`, so we are providing a specific error message.
            raise ValueError(
                f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
                f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
            )
        elif not isinstance(lm, BaseLM):
            raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

        # If temperature is unset or <=0.15, and n > 1, set temperature to 0.7 to keep randomness.
        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if "prediction" in inputs:
            if (
                isinstance(inputs["prediction"], dict)
                and inputs["prediction"].get("type") == "content"
                and "content" in inputs["prediction"]
            ):
                # If the `prediction` is the standard predicted outputs format
                # (https://platform.openai.com/docs/guides/predicted-outputs), we remove it from input kwargs and add it
                # to the lm kwargs.
                config["prediction"] = inputs.pop("prediction")

        # Populate default values for missing input fields.
        for k, v in signature.input_fields.items():
            if k not in inputs and v.default is not PydanticUndefined:
                inputs[k] = v.default

        if not all(k in inputs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in inputs]
            missing = [k for k in signature.input_fields if k not in inputs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, signature, demos, inputs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None and settings.max_trace_size > 0:
            trace = settings.trace
            if len(trace) >= settings.max_trace_size:
                trace.pop(0)
            trace.append((self, {**kwargs}, pred))
        return pred

    def _should_stream(self):
        stream_listeners = settings.stream_listeners or []
        should_stream = settings.send_stream is not None
        if should_stream and len(stream_listeners) > 0:
            should_stream = any(stream_listener.predict == self for stream_listener in stream_listeners)

        return should_stream

    def forward(
        self,
        *,
        config: dict | None = None,
        signature: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        """Execute the LM call. Override this in subclasses.

        Most callers should use `predict(...)` (that is, `__call__`)
        rather than calling `forward` directly.

        Four reserved keyword arguments receive special treatment:

        - `signature`: Replace this predictor's signature for one call.
        - `demos`: Replace the few-shot demos for one call.
        - `config`: Override LM settings such as `temperature`.
        - `lm`: Use a different language model for one call.

        Returns:
            (Prediction): Attributes match the signature's output fields.
        """
        lm, config, signature, demos, inputs = self._forward_preprocess(
            config=config,
            signature=signature,
            demos=demos,
            lm=lm,
            **inputs,
        )

        adapter = settings.adapter or ChatAdapter()

        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)
        else:
            with settings.context(send_stream=None):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)

        return self._forward_postprocess(completions, signature, **inputs)

    async def aforward(
        self,
        *,
        config: dict | None = None,
        signature: str | type[Signature] | None = None,
        demos: list | None = None,
        lm: BaseLM | None = None,
        **inputs,
    ):
        """Async version of `forward`. Same arguments and return type."""
        lm, config, signature, demos, inputs = self._forward_preprocess(
            config=config,
            signature=signature,
            demos=demos,
            lm=lm,
            **inputs,
        )

        adapter = settings.adapter or ChatAdapter()
        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)
        else:
            with settings.context(send_stream=None):
                completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=inputs)

        return self._forward_postprocess(completions, signature, **inputs)

    def update_config(self, **kwargs):
        """Merge keyword arguments into the default LM config.

        Existing keys are overwritten; new keys are added.

        Examples:
            >>> predict = Predict("q -> a", temperature=0.5)
            >>> predict.update_config(temperature=0.9, max_tokens=100)
            >>> predict.get_config()
            {'temperature': 0.9, 'max_tokens': 100}
        """
        self.config = {**self.config, **kwargs}

    def get_config(self):
        """Return the current default LM config dict."""
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"


def serialize_object(obj):
    """Recursively convert an object to a JSON-serializable form.

    Pydantic models are dumped with `model_dump(mode="json")`.
    Lists, tuples, and dicts are traversed recursively.  Primitives
    pass through unchanged.

    Args:
        obj: Any Python object.

    Returns:
        A JSON-serializable equivalent of *obj*.

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        >>> serialize_object(User(name="Ada"))
        {'name': 'Ada'}
        >>> serialize_object([1, {"key": User(name="Bob")}])
        [1, {'key': {'name': 'Bob'}}]
    """
    if isinstance(obj, BaseModel):
        # Use model_dump with mode="json" to ensure all fields (including HttpUrl, datetime, etc.)
        # are converted to JSON-serializable types (strings)
        return obj.model_dump(mode="json")
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    else:
        return obj


# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
