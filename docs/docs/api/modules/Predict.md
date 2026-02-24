# dspy.Predict

`dspy.Predict` is the fundamental building block of every DSPy program. It takes a *signature* — a declarative specification of input and output fields — and calls the configured language model to produce the outputs. All other DSPy modules (`ChainOfThought`, `ReAct`, etc.) are built on top of `Predict`.

<!-- START_API_REF -->
::: dspy.Predict
    handler: python
    options:
        members:
            - __call__
            - acall
            - aforward
            - batch
            - deepcopy
            - dump_state
            - forward
            - get_config
            - get_lm
            - inspect_history
            - load
            - load_state
            - map_named_predictors
            - named_parameters
            - named_predictors
            - named_sub_modules
            - parameters
            - predictors
            - reset
            - reset_copy
            - save
            - set_lm
            - update_config
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## Usage Examples

### Basic string signature

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

predict = dspy.Predict("question -> answer")
result = predict(question="What is the capital of France?")
print(result.answer)
```

### Typed signature class

```python
class Translate(dspy.Signature):
    """Translate the sentence to French."""
    sentence: str = dspy.InputField()
    translation: str = dspy.OutputField()

translate = dspy.Predict(Translate)
result = translate(sentence="Hello, world!")
print(result.translation)
```

### Overriding config per-call

```python
predict = dspy.Predict("question -> answer", temperature=0.0)

# Override temperature for just this call
result = predict(question="What is 2+2?", config={"temperature": 0.9})
```

### Multiple completions

```python
predict = dspy.Predict("question -> answer", n=3, temperature=0.7)
result = predict(question="Name a programming language.")

# Access all completions
for completion in result.completions:
    print(completion.answer)
```
