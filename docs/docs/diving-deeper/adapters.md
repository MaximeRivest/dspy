# Adapters: how signatures become prompts

## Intent

An adapter is the layer between a `Signature` and the LM. It renders the signature's instructions, fields, and demos into chat messages, and parses the LM's response back into typed Python values. Different adapters use different prompt shapes — chat markers, JSON, XML, or a two-stage extract — so the same Signature can run against models with widely different formatting strengths.

Read this when you want to know what the prompt looks like on the wire, why a typed field parses one way and not another, where the `[[ ## field_name ## ]]` markers come from, or which adapter to switch to when ChatAdapter is misbehaving.

## The one line to hold onto

Your signature is **intent**: what you asked for. Everything the adapter does is **mechanism**: how it gets delivered. You can swap any part of the mechanism — the adapter, how a field is served, how a value is encoded — and your task has not changed: same signature, same program, same metric. That separation is what makes adapters swappable, and it is enforced, not aspirational: a prediction contains exactly your declared output fields, and nothing in the adapter layer ever makes an extra LM call.

## Design decisions

### 1. Adapters are pluggable

Same Signature, different prompt shape, no Signature changes. LMs vary a lot in what they prefer to read and produce: a model trained on instruction-following with markers loves ChatAdapter's `[[ ## field ## ]]` format; a model with native structured-output mode does best with JSONAdapter; a reasoning model that formats unreliably wants TwoStepAdapter. Putting this decision behind one interface means Signatures, modules, and optimizers don't have to know which family of LM they're running against.

### 2. ChatAdapter is the default

Text-only, model-agnostic, uses `[[ ## field ## ]]` markers. It needs no special LM features — no JSON mode, no function calling, no native structured outputs. That breadth is why it's the default. It also includes a safety net: when its regex parser fails, it falls back to JSONAdapter automatically (toggle with `use_json_adapter_fallback=False`).

### 3. One call flows through a plan

Internally, every adapter call follows the same pipeline: **plan → render → one LM call → parse**. Planning decides everything up front — which fields the model will see, which native LM features will serve which fields, what gets stripped from or added to the request — before any text is generated. Rendering turns that plan into messages. Parsing recovers your typed fields from the response, using the same plan.

Two properties of this pipeline are guarantees you can rely on:

- **Planning is pure.** No LM is called during planning or rendering, so you can inspect what *would* be sent without spending a token.
- **One adapter call is one LM exchange.** Nothing inside the adapter's rendering or parsing machinery makes an extra LM call. The visible exceptions are explicit, named behaviors that wrap the exchange: ChatAdapter's JSON fallback retries the request once under a different format (decision 7), and TwoStepAdapter is openly a two-call design (decision 9).

### 4. Type coercion is centralized

`dspy/adapters/utils.py::parse_value` is the single function every adapter delegates to. Adapters differ in how they *find* a field's value in the LM's output (regex marker, JSON key, XML tag), but once they have the raw string they all call the same `parse_value(value, annotation)`. That keeps coercion rules consistent across adapter swaps and gives you one place to look when a typed field misbehaves.

### 5. A field is a shape plus a role

A signature field carries two independent things:

- Its **shape** — the data type: `str`, `int`, a pydantic model, `list[Quote]`, an enum. Any type with a JSON-schema meaning works; you do not need special DSPy types for data. (Types with no JSON-schema meaning, like `Callable`, are refused with an error naming the field — a callable is a tool, not data.)
- Its **semantic role** — what the field means to the exchange: is it ordinary data (`plain`), the model's thinking channel (`reasoning`), invokable capabilities (`tools`), grounded citations (`citations`), conversation turns (`history`), an image or audio input (`media`)?

The role vocabulary is small and closed: `plain`, `reasoning`, `tools`, `tool_calls`, `citations`, `history`, `media`, `code`. Roles are declared in the signature — they are part of your intent — while the machinery that *serves* a role (a native provider feature, or a textual pattern in the prompt) belongs to the adapter layer and can change without touching your program.

You can declare a role three ways today, all resolving to the same thing:

```python
from typing import Annotated
from dspy.signatures.roles import citations

# 1. Canonical — type checkers see `str`, dspy sees the role:
answer: Annotated[str, citations] = dspy.OutputField()

# 2. Subscript sugar — identical to the Annotated form:
answer: citations[str] = dspy.OutputField()

# 3. Keyword — validated eagerly, unknown roles raise immediately:
answer: str = dspy.OutputField(role="citations")
```

The `Annotated` form also works inside string signatures (pass the marker via `custom_types`). Conflicting declarations on one field — say `role="citations"` on a `Reasoning`-typed field — raise immediately rather than guessing.

The familiar semantic types still work and are not deprecated: `dspy.Reasoning` is exactly the fused spelling of "shape `str` + role `reasoning`", and every legacy type implies its role automatically (`Tool` → `tools`, `Citations` → `citations`, `Image`/`Audio`/`File`/`Document` → `media`, `History` → `history`, `Code` → `code`). The split form exists because roles and shapes vary independently: `citations[list[Quote]]` — structured, grounded quote objects — is expressible in one line, where a fused type would have to pin both halves at once.

### 6. Native LM features are served per role, capability-checked

Function calling, native reasoning modes, and provider citations are handled by the engine per role: when planning a call, each role-carrying field is matched against what the configured LM actually supports, and the plan records what was chosen (use the native channel, or render the request textually into the prompt). A model with native function calling gets a real `tools` array in the request; a model without it gets tool specs rendered as text and calls parsed from the response. Your signature is identical in both cases — that is the point of declaring the role rather than the mechanism.

Custom `dspy.Type` subclasses that implement the documented `adapt_to_native_lm_feature()` / `parse_lm_response()` hooks continue to work unchanged; the engine wraps them so they participate in planning like the built-ins do.

### 7. ChatAdapter falls back to JSONAdapter on parse error

Toggleable; on by default. The first time the LM produces malformed `[[ ## ## ]]` output, ChatAdapter catches the parse error and re-runs the request through JSONAdapter. This is more lenient than the right behavior for new code — you'd prefer to see the error — but it's the default because it makes ChatAdapter usable against a wider range of models out of the box. Set `use_json_adapter_fallback=False` in tests. Note this is a deliberate exception to the one-call rule in decision 3: a fallback is a second, recorded exchange, not hidden machinery.

### 8. JSONAdapter prefers structured output mode

It checks `lm.supported_params` and falls through tiers: OpenAI-style `response_format: json_schema` first, then `json_object` mode, then plain text JSON. The first tier is the most reliable because the model is constrained at decoding time, not only instructed.

### 9. TwoStepAdapter splits generation from extraction

For reasoning models that produce free-form text but format unreliably. Stage one: the main LM gets the task as plain prose, no field markers, and produces whatever shape it wants. Stage two: a smaller extractor LM with ChatAdapter reads the output and pulls out the declared fields. Useful for o1, o3-mini, and similar models where formatting reliability is the bottleneck. It is openly a two-call design — you configure the extraction model explicitly.

### 10. How a value is written is a codec, not an adapter — and BAML is a codec

When a field holds an object — a pydantic model, a list, a dict — someone decides how to *write it into* the prompt and what syntax to ask for back. That decision is a **codec**: a render/parse pair for values, separate from the adapter's overall prompt shape.

BAML is the ratified example: it is **not a peer prompt shape but a codec** — indented, readable pydantic rendering on the input side plus a compact schema presentation the model follows more easily for deeply nested types. The `BAMLAdapter` class is simply today's packaging of "the JSON prompt shape with the BAML codec bound"; as the preset system lands (see *Arriving* below) it becomes a compatibility shim over exactly that pairing. Same exchange structure, different value encoding.

Today the codec pairing is fixed per adapter. The named concept matters anyway, because it tells you where to look when a rendered value surprises you: the adapter decides the message structure; the codec decides how your object appears inside it.

### 11. Field marker format is hard-coded

`[[ ## name ## ]]` is the pattern, chosen for low collision and clean regex. The brackets-plus-hashes shape is unlikely to appear in real text or code, and the symmetry makes the parser simple. There's no config knob to change it. JSONAdapter and XMLAdapter use their own formats; if you want a different prompt shape entirely, that's what templates are for (see *Arriving* below).

### 12. Predictions carry exactly what you declared

A prediction's fields are your signature's output fields — no more, no less. Anything else a module generates while answering (a chain of thought you did not ask for, a tool trajectory, REPL history) is **mechanism exhaust**, and it lives in a separate observability channel: `prediction._trajectory`, a dict you can read for debugging. Exhaust never appears in `prediction.keys()`, item access, or serialized state.

For compatibility, attribute access still reaches the channel: `result.reasoning` on a ChainOfThought prediction returns the reasoning with a `DeprecationWarning` pointing you at `_trajectory`. The rule for making it contractual is one sentence: **want it? declare it.** Put `reasoning: str = dspy.OutputField()` (or a `reasoning`-role field) in your signature and it is a real output — present in the prediction, no warning, yours.

### 13. Finetuning data export is per-adapter

`format_finetune_data` is implemented on ChatAdapter (OpenAI message format); JSONAdapter raises `NotImplementedError`; TwoStepAdapter doesn't support it either. If you're using `BootstrapFinetune`, stay on ChatAdapter or implement `format_finetune_data` on the adapter of your choice.

## API walkthrough

Grouped by what you're trying to do.

### The adapters

**`dspy.ChatAdapter(callbacks=None, use_native_function_calling=False, native_response_types=None, use_json_adapter_fallback=True)`**  
The default. Builds a chat-style prompt with field markers, parses the response with a regex over the same markers, and (by default) falls back to JSONAdapter if the regex misses. Set `use_json_adapter_fallback=False` when you want a hard error in tests.

**`dspy.JSONAdapter(callbacks=None, use_native_function_calling=True)`**  
Outputs structured JSON. Formatting is similar to ChatAdapter on the input side, but the output instruction asks for JSON and parsing uses `json_repair`. The constructor's `use_native_function_calling=True` default flips when tool calling is wired in.

**`dspy.XMLAdapter(callbacks=None)`**  
`<field_name>value</field_name>` tags. The parser is a regex (`r"<(\w+)>(.*?)</\1>"` with `DOTALL`); it's robust to whitespace but doesn't tolerate nested tags of the same name.

**`dspy.TwoStepAdapter(extraction_model: BaseLM, **kwargs)`**  
Two LM calls per inference. Use it when the main LM is a reasoning model that's bad at formatting — the extractor is usually a cheap general-purpose LM with ChatAdapter. Doesn't support finetuning yet.

**`BAMLAdapter`** (import from `dspy.adapters.baml_adapter`)  
The JSON prompt shape with the BAML *codec* bound: pydantic inputs render as indented JSON, and the output schema is presented in BAML-style commented-Pydantic form. Worth trying when JSONAdapter's raw JSON schema is too verbose for complex nested types. Per decision 10, BAML is a codec, not a prompt shape — this class is the current packaging of that pairing and will become a compatibility shim once codecs are independently bindable.

**`dspy.Adapter`**  
The base class every adapter extends. Subclassing it remains possible (implement `format()`, `parse()`, and optionally `format_finetune_data()`), but for new prompt shapes prefer waiting for templates — a subclass hand-maintains what a template declares as data (see *Customizing the prompt* above).

### The call path, and customizing the prompt

**`Adapter.__call__(lm, lm_kwargs, signature, demos, inputs)` / `Adapter.acall(...)`**  
Public entry. The flow inside is plan → render → LM call → parse (decision 3). `lm_kwargs` (temperature, max_tokens, response_format, etc.) is where planning records requests for structured output or function calling.

**`Adapter.format(signature, demos, inputs)` → `list[dict]`**  
Renders the signature, demos, and inputs into the chat messages that would be sent — call it directly to inspect the prompt without spending a token.

**`Adapter.parse(signature, completion)` → `dict`**  
Extracts typed field values from the LM's response. ChatAdapter regex-matches the marker pattern, splits by field, and delegates each value to `parse_value`. JSONAdapter parses the JSON object and pulls values by key. XMLAdapter walks tags.

**`Adapter.format_finetune_data(signature, demos, inputs, outputs)`**  
Serializes a demo into the LM provider's finetune format. ChatAdapter writes OpenAI message format. Other adapters raise `NotImplementedError`.

**Customizing the prompt.** The intended authoring surface for "I want the messages to look exactly like *this*" is the **template**: your prompt as a literal message list with interpolation slots, arriving with the preset system (see *Arriving* below). Until it lands, know that the built-in adapters compose their prompts from a family of `format_*` helper methods (`format_system_message`, `format_field_description`, `format_demos`, …) that you can technically override in a subclass. They still work and existing subclasses keep working — but this is the legacy customization surface, scheduled for deprecation once presets land, so don't build new prompt customization on it. If you need a different prompt *today* and can't wait, overriding `format()` wholesale is the least entangled option.

### Type coercion

The single function every adapter calls when turning an LM-produced string into a typed value.

**`parse_value(value, annotation)` in `dspy/adapters/utils.py`**  
Strategy: if the annotation is `str`, pass through. If it's an `Enum` or `Literal`, match against allowed values. Otherwise: try `json_repair.loads`, fall back to `ast.literal_eval`, fall back to the raw string; then validate with `TypeAdapter(annotation)`. If validation fails and the annotation is a `dspy.Type` subclass, retry with the raw value so the type's own parser can take a shot.

Other helpers in the same file you'll see in tracebacks:

- `format_field_value(field_info, value, assume_text=True)` — the inverse of parse: serializes a typed value for the prompt.
- `serialize_for_json(value)` — Pydantic-aware JSON serialization, used by JSONAdapter.
- `translate_field_type(field_info)` — generates the constraint string the prompt shows ("greater than: 0").
- `get_field_description_string(fields)` — the field-list rendering.
- `find_enum_member(enum_cls, raw)` — resolves an enum by name or value.

### Semantic role markers

**`dspy.signatures.roles`** — the role marker objects: `plain`, `reasoning`, `tools`, `tool_calls`, `citations`, `history`, `media`, `code`. Each is usable as `Annotated` metadata or via subscript sugar (`citations[str]` is exactly `Annotated[str, citations]`). Constructing a `SemanticRole` with an unknown name raises immediately, listing the vocabulary. See decision 5 for the three declaration spellings and the conflict rules.

### Custom type wrappers

Types adapters know how to render and parse beyond Python's standard ones. Each is the fused spelling of a shape plus a semantic role (decision 5); each implements `format()`, and some implement native-feature hooks that the engine wraps into its planning (decision 6).

**`dspy.adapters.types.Type`**  
The base class. Subclass it (it's a `pydantic.BaseModel`) and implement `format()` to plug in a new type. Adapters wrap the output of `format()` with `<<CUSTOM-TYPE-START-IDENTIFIER>>...<<END-IDENTIFIER>>` so multi-modal content can be inserted into a single message stream and later split out.

**`dspy.Image(source)`**

URL reference, data URI, bytes, or PIL image. `format()` returns the provider's image content block (`{"type": "image_url", "image_url": {"url": ...}}`). Ordinary construction and adapter parsing never access the filesystem or network. Use `Image.from_path(path)` to read a local file or `Image.from_url(url)` to download and base64-encode a remote resource. The deprecated direct call `Image(url, download=True)` also downloads for compatibility through 3.3; migrate it to `Image.from_url(url)`.

**`dspy.Audio(source)`**

A data URI, in-memory bytes, or array data; raw base64 must be passed as `Audio(data=..., audio_format=...)`. Renders as the provider's audio content block. Use `Audio.from_path(path)` or `Audio.from_url(url)` for resource loading.

**`dspy.File(file_data=None, file_id=None, filename=None)`**

Either in-memory bytes, a data URI, or a file ID (some providers preupload files and reference them by ID). Use `File.from_path(path)` to read a local file.

**`dspy.Code(code, language="python")`**  
Code with a class-level `language` parameter. `dspy.Code["java"]` produces a Code subclass typed for Java. `format()` returns the raw string — no wrapper, no fencing.

**`dspy.History(messages)`**  
Conversation turns. When the adapter sees this type on a field, it expands the messages into real user/assistant messages instead of stuffing them into one field's value. Use this when you want the LM to see prior turns as messages.

**`dspy.Reasoning(content)`**  
String-like wrapper — the fused spelling of shape `str` + role `reasoning`. When the LM supports a native reasoning mode (o1, o3-mini, GPT-5 thinking variants), planning sets `reasoning_effort` and reads the reasoning from the native response channel; otherwise the field renders and parses as ordinary text. Either way your signature is unchanged.

**`dspy.Tool(func, name=None, desc=None, args=None, arg_types=None, arg_desc=None)`**  
Wraps a Python callable. Auto-introspects the function signature if you don't pass `args`/`arg_types`/`arg_desc`. Used by `ReAct` and anywhere modules accept tools. The full tool story lives in the Tools / ReAct / MCP DD page.

**`dspy.adapters.types.tool.ToolCalls.from_dict_list(...)`**  
The list of tool calls the LM produced, parsed from native function-calling responses.

**`dspy.adapters.types.Citations`**  
Declared as a default native response type. When the provider returns citations natively (e.g., Anthropic), the engine's citations handling extracts them; on other providers the field behaves textually.

### Migrating resource loading in 3.3

In DSPy 3.3, constructing or validating `Image`, `Audio`, and `File` values no longer interprets locator-shaped strings as instructions to read a local file or fetch a remote URL. This keeps LM output parsing and other validation paths from implicitly granting access to the host. Resource loading now requires an explicit factory:

| Before 3.3 | 3.3 replacement | Behavior |
| --- | --- | --- |
| `Image(path)` | `Image.from_path(path)` | Read and embed a local image |
| `Image(url, download=True)` | `Image.from_url(url)` | Download and embed a remote image |
| `Image.from_url(url)` or `Image.from_url(url, download=False)` | `Image(url)` | Keep a non-downloading URL reference |
| `Audio(path)` | `Audio.from_path(path)` | Read and embed a local audio file |
| `Audio(url)` | `Audio.from_url(url)` | Download and embed remote audio |
| `File(path)` | `File.from_path(path)` | Read and embed a local file |
| `Image.from_file(path)` | `Image.from_path(path)` | Replace the deprecated alias |
| `Audio.from_file(path)` | `Audio.from_path(path)` | Replace the deprecated alias |

Safe in-memory inputs such as data URIs, bytes, PIL images, audio arrays, and structured dictionaries remain supported. The deprecated direct call `Image(url, download=True)` continues to work with a warning through 3.3; `Image.from_file()`, `Image.from_PIL()`, and `Audio.from_file()` are also scheduled for removal in 3.4.

`Image.from_url()` and `Audio.from_url()` perform synchronous, caller-initiated HTTP requests and follow redirects. They do not validate destinations against an SSRF allowlist, so applications must validate or allowlist URLs derived from untrusted input before calling them.

### Configuring which adapter to use

- **`dspy.configure(adapter=dspy.JSONAdapter())`** — process-wide default.
- **`with dspy.context(adapter=dspy.XMLAdapter()): ...`** — scoped override.
- **No automatic LM-based selection.** ChatAdapter is the default and stays the default until you set otherwise. Some teleprompts (e.g., `BootstrapFinetune`) accept an `adapter` dict keyed by LM, so different LMs in a finetuning loop can use different adapters.

## Arriving: presets and templates

This section describes ratified, in-progress work — none of it is callable today. It is documented here so the direction is public and so the pieces above make sense as parts of a whole.

The destination: **an adapter is a preset** — a named data entry bundling five things: a *template* (the prompt shape), a *parser binding*, *codec bindings* (decision 10), *strategy bindings* (decision 6), and resolved config. The built-in adapters become thin constructors over presets named `chat`, `json`, and `xml`; classes stay as the convenient handles, data becomes the truth.

**The template is the piece that changes how you'll customize prompts.** Instead of subclassing and overriding methods, you write your prompt as a literal message list with interpolation slots — the signature stays the I/O contract, the template decides exactly what the model sees:

```python
# Arriving — illustrative, not callable today.
adapter = TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are a concise assistant. {instruction}"},
        {"role": "user", "content": "Summarize:\n\n{text}"},
    ],
    parse_mode="full_text",
)
```

`{instruction}` interpolates the signature docstring, `{text}` the input field, and the full response maps to the output field — nothing added that you didn't write. Richer slots cover the general case: `{inputs(style=...)}` / `{outputs(style=...)}` render field blocks through a codec style, `{demos()}` and `{history()}` expand few-shot examples and conversation turns into real message pairs, and constrained loop blocks iterate fields with their names, types, and markers. The language is deliberately small — slots, loops, directives, not general templating — so a prompt shape stays analyzable, diffable data. The built-in `chat` prompt shape is expressible in it exactly, which is the proof the design rests on.

The rest of the arriving surface rides on presets:

- **Per-role strategy bindings.** `strategies={"reasoning": "native_channel", "tools": "textual_json"}` — choose *how each role is served*, with an `"auto"` default resolving against the LM's declared capabilities. Templates are strategy-aware: a natively-served role simply renders no block.
- **A named codec pool.** Register value codecs and bind them per field, including asymmetric pairs (render inputs as compact Python literals, request outputs as JSON). BAML becomes an ordinary named codec, bindable to any preset.
- **The `@role` string-signature shorthand.** `"question -> answer: str @citations"` — the fourth spelling of decision 5, plus a public `dspy.roles` import path.
- **Preset serialization.** A preset dumps to data — template, bindings, config — and reconstructs from that data alone: the foundation for saving programs whose exact prompt behavior travels with them.

## Cross-links

- [Signatures in depth](signatures-in-depth.md) — what the adapter consumes.
- [Settings and context()](settings-and-context.md) — how `configure` and `context` propagate the adapter choice.
- Tools, ReAct, and MCP DD page — `Tool` and `ToolCalls` are adapter-formatted but module-driven.
