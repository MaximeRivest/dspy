import textwrap
from pathlib import Path
from types import SimpleNamespace


def _anthropic_blocks() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/docs/migration/lm-migration.md").read_text()
    section = text.split("## Anthropic modality mapping examples", 1)[1].split(
        "## OpenAI-compatible endpoints", 1
    )[0]
    blocks = []
    in_block = False
    current = []
    for line in section.splitlines():
        if line.startswith("        ```python"):
            in_block = True
            current = []
            continue
        if in_block and line.startswith("        ```"):
            blocks.append(textwrap.dedent("\n".join(current)))
            in_block = False
            continue
        if in_block:
            current.append(line[8:] if line.startswith("        ") else line)
    return blocks


class FakeMessages:
    def create(self, **kwargs):
        content = [SimpleNamespace(type="text", text="hello")]
        if kwargs.get("thinking"):
            content = [
                SimpleNamespace(type="thinking", thinking="brief thought"),
                SimpleNamespace(type="text", text="answer"),
            ]
        elif kwargs.get("tools"):
            tool = kwargs["tools"][0]
            content = [
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name=tool.get("name", "tool"),
                    input={"city": "Paris"},
                )
            ]
        return SimpleNamespace(
            model=kwargs["model"],
            content=content,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    def stream(self, **kwargs):
        return FakeStream()


class FakeStream:
    def __enter__(self):
        return iter(
            [
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="hello"),
                )
            ]
        )

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = FakeMessages()


def test_anthropic_migration_examples_compile_and_run(monkeypatch, tmp_path):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dog.png").write_bytes(b"fake image bytes")

    namespace = {}
    blocks = _anthropic_blocks()

    assert len(blocks) == 7
    for index, block in enumerate(blocks, start=1):
        compile(block, f"anthropic-migration-block-{index}", "exec")
        exec(block, namespace)

    lm = namespace["AnthropicCitationLM"]("claude-sonnet-4-5-20250929", cache=False)
    request = lm.normalize_request(
        namespace["dspy"].User(
            "Use this document.",
            namespace["dspy"].LMFilePart(
                data="ZG9j",
                media_type="application/pdf",
                filename="paper.pdf",
            ),
        )
    )
    provider_request = lm._request_kwargs(request)

    assert provider_request["messages"][0]["content"][1]["type"] == "document"
    assert provider_request["messages"][0]["content"][1]["citations"] == {"enabled": True}
