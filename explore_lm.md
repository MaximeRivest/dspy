# Getting started with the LM

A walkthrough of the LM layer, from "hello" to a full program. Run the
cells in order. All outputs below are real runs against
`openai-codex:gpt-5.6-luna`.

One idea to hold on to: every call builds one canonical **request**,
and the model sends back one canonical **response**. Everything below
is a different way to build that request or read that response.

## 1. Say hello

Construct an LM from a model string — a bare family name
(`"gpt-4o-mini"`) or an explicit `provider:` prefix
(`"openai-codex:gpt-5.6-luna"`, `"ollama:llama3.2"`). The router picks
up the provider's API key from its usual environment variable. Then
call it with a string:

```python
import dspy

lm = dspy.LM("openai-codex:gpt-5.6-luna")
response = lm("hello")
response
```
```output | ✓ 1.5s | 12 vars
Response(
    text='Hello! How can I help you today?',
    model='gpt-5.6-luna',
    finish_reason='stop',
    usage=Usage(input_tokens=17, output_tokens=13, total_tokens=30, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0360b809c0c4ad02016a7fab5cacd88190b5a2188d59858f41',
    provider_data=<dict: 35 keys>,
)
```

You get a rich `Response`, not a bare string. The parts you will use
most:

```python
response.text
```
```output | ✓ 22ms | 12 vars
'Hello! How can I help you today?'
```

```python
response.usage.total_tokens
```
```output | ✓ 23ms | 12 vars
30
```

There is also `response.tool_calls` and `response.citations` — data
when the model uses those channels, empty otherwise.

## 2. Build a conversation

A conversation is a list of turns, spelled with constructors named
after the speaker: `dspy.System` (instructions), `dspy.User` (you),
`dspy.Assistant` (the model, in earlier turns). And a previous
`Response` drops straight in as its own turn:

```python
first = lm(dspy.User("What is DSPy?"))
print(first.text[:130] + " ...")
```
```output | ✓ 6.8s | 12 vars
**DSPy** is a Python framework for building and optimizing applications that use language models.

Instead of manually writing and ...
```

The model wrote a whole essay. Feed it back and ask for less:

```python
follow = lm(
    dspy.System("Answer in five words or less."),
    dspy.User("What is DSPy?"),
    first,
    dspy.User("Say it shorter."),
)
print(follow.text)
```
```output | ✓ 1.0s | 12 vars
DSPy optimizes LLM programs automatically.
```

That is the whole conversation API: pass the turns in order, get a
`Response` back.

## 3. Show the model a tool exchange

Two more constructors describe tool use: `dspy.ToolCall` (the model
asked to run a tool) and `dspy.ToolResult` (what the tool returned).
Here we replay a finished tool exchange and ask for a summary:

```python
answer = lm(
    dspy.User("What is the weather in Paris?"),
    dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
    dspy.ToolResult('{"temperature": "22 C", "sky": "sunny"}', call_id="call_1", name="get_weather"),
    dspy.User("Summarize the result."),
)
print(answer.text)
```
```output | ✓ 1.7s | 12 vars
Paris is sunny with a temperature of 22°C.
```

## 4. Stream the answer

`lm.stream(...)` takes the same inputs as `lm(...)`, but you get the
text piece by piece while the model writes it:

```python
stream = lm.stream("Write a haiku about rivers.")
for text in stream:
    print(text, end="", flush=True)
```
```output | ✓ 1.6s | 12 vars
River stones whisper  
Moonlight drifts along the current  
Willows bend and breathe
```

After the loop, the finished `Response` is right there — the same
object a non-streamed call returns:

```python
stream.response.usage
```
```output | ✓ 23ms | 12 vars
Usage(input_tokens=23, output_tokens=21, total_tokens=44, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None)
```

Under the text there is a typed event stream — start, deltas, end.
You rarely need it, but it is one `.events()` away:

```python
for event in lm.stream("Count from 1 to 5, one number per line.").events():
    print(event)
```
```output | ✓ 1.6s | 12 vars
StreamStartEvent(id='resp_0b871f119365cbcc016a7fab83ced88194a38bd2d6ba823f09', model='gpt-5.6-luna', type='start')
StreamDeltaEvent(delta=TextDelta(text='1', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='2', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='3', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='4', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'), type='delta')
StreamDeltaEvent(delta=TextDelta(text='5', part_index=0, type='text'), type='delta')
StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=29, output_tokens=13, total_tokens=42, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 35 keys>,
    type='end',
)
```

## 5. The plain-strings call

There is a second, older way to call the LM: keyword arguments
(`prompt=` or `messages=`) in, a list of strings out. This is the face
DSPy's adapters use internally. You will mostly read it in their code,
not write it:

```python
lm(prompt="hello")
```
```output | ✓ 978ms | 12 vars
['Hello! How can I help you today?']
```

The rule of thumb: positional inputs → rich `Response`; keyword
`prompt=`/`messages=` → plain list of strings.

## 6. From calls to programs

Direct calls are for exploring. The real DSPy move is to declare *what*
you want — a `Signature` — and let a program handle the prompting.
`dspy.configure(lm=...)` tells programs which model to use:

```python
class QA(dspy.Signature):
    """Answer in one short sentence, for a curious teenager."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

dspy.configure(lm=lm)
program = dspy.Predict(QA)

prediction = program(question="Why is the sky blue?")
print(prediction.answer)
```
```output | ✓ 4.3s | 12 vars
The sky looks blue because air molecules scatter blue sunlight more strongly than other colors, sending more blue light into our eyes from every direction.
```

## 7. See the exact prompt

Every call is recorded in `lm.history`. Look at the last one to see
exactly what the program put on the wire — the adapter turned your
signature into instructions and markers:

```python
lm.history[-1]["messages"]
```
```output | ✓ 23ms | 12 vars
[{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer in one short sentence, for a curious teenager.'}, {'role': 'user', 'content': '[[ ## question ## ]]\nWhy is the sky blue?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}]
```

## 8. Count tokens and cost

Every prediction carries the usage of the run that made it — always
on, no flag to set:

```python
prediction.get_lm_usage()
```
```output | ✓ 23ms | 13 vars
{'openai-codex:gpt-5.6-luna': Usage(input_tokens=140, output_tokens=41, total_tokens=181, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None)}
```

And `lm.history` remembers every call, so the whole session sums in
one line:

```python
len(lm.history), sum(x["usage"].total_tokens or 0 for x in lm.history)
```
```output | ✓ 24ms | 13 vars
(8, 1234)
```

Dollar cost rides beside it — `lm.history[-1]["cost"]` and
`prediction.get_lm_cost()` — estimated from the model catalog's
per-token prices. For this model both return `None`, and that is
correct, not missing: the codex provider bills through your ChatGPT
subscription, not per token, so there is no honest per-call price.
Catalog-priced models (for example `openai:gpt-4o-mini`) return real
dollars, and `sum(x["cost"] for x in lm.history if x["cost"])` is your
session total.

That is the full tour: call a model, hold a conversation, stream an
answer, let a program write the prompts for you — then check its work
and its bill in `lm.history`.

