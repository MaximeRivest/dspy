"""`LM.complete` rides the same engine seam as the typed face.

An `api_base`/`api_key` endpoint pin must apply to caller-built
requests too — found live when `lm.complete` sent a lab-server request
to api.openai.com (tests/live/test_vllm.py, tool emission).
"""

from lm15 import Message, Request, Response, TextPart, Usage
from lm15.providers import OpenAIChatLM

import dspy


def _canned_response(model: str) -> Response:
    return Response(
        id=None,
        model=model,
        message=Message(role="assistant", parts=(TextPart(text="ok"),)),
        finish_reason="stop",
        usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def test_complete_honors_pinned_endpoint(monkeypatch):
    lm = dspy.LM("openai-chat:gemma/big", api_base="http://lab.test/v1", api_key="k")
    seen = {}

    def fake_complete(engine, request):
        seen["engine"] = engine
        seen["model"] = request.model
        return _canned_response(request.model)

    monkeypatch.setattr(lm, "_complete", fake_complete)
    lm.complete(Request(model="openai-chat:gemma/big", messages=(Message.user("hi"),)))

    assert seen["model"] == "gemma/big"  # prefix never leaks to the wire
    assert isinstance(seen["engine"], OpenAIChatLM)
    assert seen["engine"].base_url == "http://lab.test/v1"


def test_complete_leaves_foreign_model_strings_alone(monkeypatch):
    lm = dspy.LM("openai-chat:gemma/big", api_base="http://lab.test/v1", api_key="k")
    seen = {}

    def fake_complete(engine, request):
        seen["model"] = request.model
        return _canned_response(request.model)

    monkeypatch.setattr(lm, "_complete", fake_complete)
    lm.complete(Request(model="something/else", messages=(Message.user("hi"),)))

    assert seen["model"] == "something/else"


def test_complete_without_endpoint_uses_the_router(monkeypatch):
    class _Router:
        def complete(self, request):
            return _canned_response(request.model)

    router = _Router()
    lm = dspy.LM("openai-chat:stub-model", router=router)
    seen = {}

    def fake_complete(engine, request):
        seen["engine"] = engine
        seen["model"] = request.model
        return _canned_response(request.model)

    monkeypatch.setattr(lm, "_complete", fake_complete)
    lm.complete(Request(model="openai-chat:stub-model", messages=(Message.user("hi"),)))

    assert seen["engine"] is router  # explicit router always wins
    assert seen["model"] == "openai-chat:stub-model"
