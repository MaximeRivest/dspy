# Getting started with the LM

A walkthrough of the LM layer, from "hello" to a full program. Run each
cell in order. All cells use `DummyLM` — a fake model that replays
answers you script — so you need no network and no API keys.

One idea to hold on to: every call builds one canonical **request**, and
the model sends back one canonical **response**. Everything below is a
different way to build that request or read that response.

## 1. Say hello

A `DummyLM` takes a list of answers. Each call consumes the next one.
Call it with a string, and you get a `Response` back.

```python
import dspy

lm = dspy.DummyLM([
    "Hello! Nice to meet you.",
    "DSPy is a way to program language models instead of prompting them.",
    "Programming, not prompting.",
    "Hello again!",
])

response = lm("hello")
response
```
```output | ✓ 2.3s | 3 vars
Response(
    text='Hello! Nice to meet you.',
    model='dummy',
    finish_reason='stop',
    usage=Usage(input_tokens=None, output_tokens=None, total_tokens=None, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
)
```

The response is a rich object, not a bare string. The parts you will
use most:

```python
response.text
```
```output | ✓ 22ms | 3 vars
'Hello! Nice to meet you.'
```

There is also `response.usage` (token counts), `response.tool_calls`,
and `response.citations` — they matter later, with real models.

## 2. Use a real model

For a real model, swap `DummyLM` for `LM` and give it a model string:

```python
lm = dspy.LM("gpt-4o-mini")            # bare family name
lm = dspy.LM("ollama:llama3.2")        # explicit provider: prefix
```

The router resolves the model string to a provider and reads that
provider's API key from its usual environment variable (for example
`OPENAI_API_KEY`). That is the whole setup. Everything else in this
walkthrough works the same on `LM` and `DummyLM`.

## 3. Build a conversation

A conversation is a list of turns. You spell each turn with a
constructor named after the speaker: `dspy.System` (instructions),
`dspy.User` (you), `dspy.Assistant` (the model, in earlier turns).
And a previous `Response` drops straight in as its own turn:

```python
first = lm(dspy.User("What is DSPy?"))

follow = lm(
    dspy.System("Answer in five words or less."),
    dspy.User("What is DSPy?"),
    first,                                # the model's earlier answer
    dspy.User("Say it shorter."),
)
print(follow.text)
```
```output | ✓ 22ms | 5 vars
Programming, not prompting.
```

That is the whole conversation API: pass the turns in order, get a
`Response` back.

## 4. Show the model a tool exchange

Two more constructors describe tool use: `dspy.ToolCall` (the model
asked to run a tool) and `dspy.ToolResult` (what the tool returned).
Here we replay a finished tool exchange and ask for a summary:

```python
weather_lm = dspy.DummyLM(["It is 22 C and sunny in Paris right now."])

answer = weather_lm(
    dspy.User("What is the weather in Paris?"),
    dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
    dspy.ToolResult('{"temperature": "22 C", "sky": "sunny"}', call_id="call_1", name="get_weather"),
    dspy.User("Summarize the result."),
)
print(answer.text)
```
```output | ✓ 23ms | 7 vars
It is 22 C and sunny in Paris right now.
```

## 5. Stream the answer

`lm.stream(...)` takes the same inputs as `lm(...)`, but you get the
text piece by piece while the model writes it. Loop over the stream to
print each piece:

```python
poet = dspy.DummyLM(["A river carries / every cloud it swallowed / back to the ocean."])

stream = poet.stream("Write a haiku about rivers.")
for text in stream:
    print(text, end="", flush=True)
```
```output | ✓ 23ms | 10 vars
A river carries / every cloud it swallowed / back to the ocean.
```

After the loop, the finished `Response` is right there — the same
object a non-streamed call returns:

```python
stream.response.text
```
```output | ✓ 22ms | 10 vars
'A river carries / every cloud it swallowed / back to the ocean.'
```

Under the text there is a typed event stream — start, deltas, end.
You rarely need it, but it is one `.events()` away:

```python
poet = dspy.DummyLM(["Snow melts on the peak / the river remembers it / all the way down."])
for event in poet.stream("Another haiku, please.").events():
    print(event)
```
```output | ✓ 22ms | 11 vars
StreamStartEvent(id=None, model='dummy', type='start')
StreamDeltaEvent(delta=TextDelta(text='Snow melts on the peak / the river remembers it / all the way down.', part_index=0, type='text'), type='delta')
StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=None, output_tokens=None, total_tokens=None, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    type='end',
)
```

## 6. The plain-strings call

There is a second, older way to call the LM: keyword arguments
(`prompt=` or `messages=`) in, a list of strings out. This is the face
DSPy's adapters use internally. You will mostly read it in their code,
not write it:

```python
lm(prompt="hello")
```
```output | ✓ 22ms | 11 vars
['Hello again!']
```

The rule of thumb: positional inputs → rich `Response`; keyword
`prompt=`/`messages=` → plain list of strings.

## 7. From calls to programs

Direct calls are for exploring. The real DSPy move is to declare *what*
you want — a `Signature` — and let a program handle the prompting.
`dspy.configure(lm=...)` tells programs which model to use:

```python
qa_lm = dspy.DummyLM([
    "[[ ## answer ## ]]\nBlue light bounces around in the air more than other colors.\n\n[[ ## completed ## ]]"
])

class QA(dspy.Signature):
    """Answer in one short sentence, for a curious teenager."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

dspy.configure(lm=qa_lm)
program = dspy.Predict(QA)

prediction = program(question="Why is the sky blue?")
print(prediction.answer)
```
```output | ✓ 31ms | 15 vars
Blue light bounces around in the air more than other colors.
```

(The scripted answer wears `[[ ## ... ## ]]` markers because that is
the wire format the default adapter asks the model to use — see for
yourself below.)

## 8. See the exact prompt

Every call is recorded in `lm.history`. Look at the last one to see
exactly what the program put on the wire — the adapter turned your
signature into instructions and markers:

```python
qa_lm.history[-1]["messages"]
```
```output | ✓ 22ms | 15 vars
[{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer in one short sentence, for a curious teenager.'}, {'role': 'user', 'content': '[[ ## question ## ]]\nWhy is the sky blue?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}]
```

That is the full tour: call a model, hold a conversation, stream an
answer, then let a program write the prompts for you — and check its
work in `history`.
