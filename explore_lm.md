# Explore the greenfield LM syntax

Run each cell in order. All cells use `DummyLM` — no network, no keys.

## Construct an LM

Model strings: bare family name (`"gpt-4o-mini"`) or explicit
`provider:` prefix (`"openai-chat:qwen3"`, `"ollama:llama3.2"`).
Capabilities are explicit boolean facts.

```python
import dspy

lm = dspy.LM("openai-codex:gpt-5.6-luna")
lm(prompt='hello')
```
```output | ✓ 907ms | 2 vars
['Hello! How can I help you today?']
```

## Call an LM directly

```python
lm(prompt='hello')
```
```output | ✓ 4.4s | 2 vars
['Hello! How can I help you today?']
```

```python
import dspy
print(dspy.__file__)
```
```output | ✓ 22ms | 2 vars
/home/maxime/Projects/dspy-greenfield/dspy/__init__.py
```

```python

class QA(dspy.Signature):
    """Answer in one short sentence."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

dspy.configure(lm=lm)

program = dspy.Predict(QA)
prediction = program(question="Why is the sky blue?")
print(prediction.answer)
```
```output | ✓ 3.6s | 6 vars
The sky appears blue because air molecules scatter short blue wavelengths of sunlight more strongly than longer wavelengths like red.
```

## 5. See the exact prompt

```python
lm.history
```
```output | ✓ 23ms | 6 vars
[{'model': 'openai-codex:gpt-5.6-luna', 'messages': [{'role': 'user', 'content': 'hello'}], 'kwargs': {'temperature': None, 'max_tokens': 4000}, 'outputs': ['Hello! How can I help you today?'], 'timestamp': 1786739123.076219}, {'model': 'openai-codex:gpt-5.6-luna', 'messages': [{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Answer in one short sentence.'}, {'role': 'user', 'content': '[[ ## question ## ]]\nWhy is the sky blue?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}], 'kwargs': {'temperature': None, 'max_tokens': 4000}, 'outputs': ['[[ ## answer ## ]]\nThe sky appears blue because air molecules scatter short blue wavelengths of sunlight more strongly than longer wavelengths like red.\n\n[[ ## completed ## ]]'], 'timestamp': 1786739178.6272957}]
```