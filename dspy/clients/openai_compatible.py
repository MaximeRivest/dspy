"""Call OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

from typing import Any

import pydantic
import requests

from dspy.clients.base_lm import BaseLM


class OpenAICompatibleLM(BaseLM):
    """Call a model served by an OpenAI-compatible chat API.

    Use this class for servers that expose `/chat/completions`, such as
    Ollama, vLLM, SGLang, LM Studio, and many hosted OpenAI-compatible
    endpoints. Pass the endpoint URL explicitly; DSPy does not guess it from
    the model name.

    Args:
        model: Model name to send in the request body. If the model starts with
            `openai-compatible:`, the prefix is removed before requests are sent.
        base_url: Base URL for the OpenAI-compatible API, for example
            `http://localhost:11434/v1`. You may also pass `api_base` for
            compatibility with existing DSPy code.
        api_key: Optional bearer token. If omitted, no authorization header is
            added.
        api_args: Extra JSON fields merged into every request body.
        api_headers: Extra HTTP headers merged into every request.
        timeout: Request timeout in seconds.
        **kwargs: Default LM parameters such as `temperature` and `max_tokens`.

    Examples:
        Ollama running locally:
        ```python
        import dspy

        lm = dspy.OpenAICompatibleLM(
            "llama3.2",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        dspy.configure(lm=lm)
        ```

        Through the `dspy.LM` resolver:
        ```python
        import dspy

        lm = dspy.LM(
            "openai-compatible:llama3.2",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        ```
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        api_args: dict[str, Any] | None = None,
        api_headers: dict[str, str] | None = None,
        timeout: float = 60,
        **kwargs,
    ):
        base_url = base_url or api_base
        if not base_url:
            raise ValueError(
                "`base_url` is required for OpenAI-compatible APIs. "
                "For OpenAI itself, use `dspy.LM('openai/<model>')`."
            )

        if model.startswith("openai-compatible:"):
            model = model.removeprefix("openai-compatible:")

        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_args = api_args or {}
        self.api_headers = api_headers or {}
        self.timeout = timeout

    @property
    def supports_function_calling(self) -> bool:
        return True

    @property
    def supports_response_schema(self) -> bool:
        return True

    @property
    def supported_params(self) -> set[str]:
        return {
            "frequency_penalty",
            "logprobs",
            "max_completion_tokens",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "temperature",
            "tools",
            "top_logprobs",
            "top_p",
        }

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": "DSPy"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.api_headers)
        return headers

    def _request_body(self, messages: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        params.pop("rollout_id", None)

        response_format = params.get("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }

        return {
            "model": self.model,
            "messages": messages,
            **params,
            **self.api_args,
        }

    def forward(self, prompt=None, messages=None, **kwargs):
        messages, params = self.prepare_request(prompt, messages, **kwargs)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=self._request_body(messages, params),
            headers=self._headers(),
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(f"{exc}: {response.text}") from exc
        return response.json()

    def dump_state(self):
        state = super().dump_state()
        state.update(
            {
                "backend": "dspy.clients.openai_compatible:OpenAICompatibleLM",
                "base_url": self.base_url,
                "api_args": self.api_args,
                "api_headers": self.api_headers,
                "timeout": self.timeout,
            }
        )
        return state
