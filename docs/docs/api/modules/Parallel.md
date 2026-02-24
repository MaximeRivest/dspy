# dspy.Parallel

`dspy.Parallel` executes multiple DSPy module calls concurrently using threads. Pass it a list of `(module, inputs)` pairs and it returns the results in the same order — handling thread isolation, error collection, progress bars, and straggler resubmission automatically.

For the common case of running **one module** over many examples, see [`dspy.Module.batch()`](https://dspy.ai/api/primitives/Module#dspy.Module.batch), which is a convenience wrapper around `Parallel`.

<!-- START_API_REF -->
::: dspy.Parallel
    handler: python
    options:
        members:
            - __call__
            - forward
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

### Running one module over many inputs

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

predict = dspy.Predict("question -> answer")
parallel = dspy.Parallel(num_threads=4)

examples = [
    dspy.Example(question="What is 1+1?").with_inputs("question"),
    dspy.Example(question="What is 2+2?").with_inputs("question"),
    dspy.Example(question="What is 3+3?").with_inputs("question"),
]

results = parallel([(predict, ex) for ex in examples])

for r in results:
    print(r.answer)
```

!!! tip
    If all your pairs use the **same module**, `module.batch(examples)` is a shorthand that does the same thing.

### Mixing different modules

`Parallel` shines when you need to fan out calls to **different** modules or signatures in one go:

```python
cot = dspy.ChainOfThought("question -> answer")
predict = dspy.Predict("question -> answer")

results = parallel([
    (cot,     dspy.Example(question="Why is the sky blue?").with_inputs("question")),
    (predict, dspy.Example(question="What is 2+2?").with_inputs("question")),
])
```

### Handling errors

Set `return_failed_examples=True` to get back the inputs that failed along with their exceptions, instead of silently returning `None` for those positions:

```python
parallel = dspy.Parallel(
    num_threads=2,
    return_failed_examples=True,
    provide_traceback=True,
)

results, failed_examples, exceptions = parallel(exec_pairs)

for ex, exc in zip(failed_examples, exceptions):
    print(f"Failed on {ex} with {exc}")
```

### Nested parallel execution

A `Parallel` instance can itself be used as the "module" in a pair, enabling nested fan-outs:

```python
inner = dspy.Parallel(num_threads=2)
outer = dspy.Parallel(num_threads=2)

results = outer([
    (predict, example1),
    (inner, [
        (predict, example2),
        (predict, example3),
    ]),
])
# results[0] is a Prediction; results[1] is a list of Predictions
```

### Input types

The second element of each pair can be:

| Type | How it is passed to the module |
|---|---|
| `dspy.Example` | `module(**example.inputs())` (or `module(example)` if `access_examples=False`) |
| `dict` | `module(**dict)` |
| `tuple` | `module(*tuple)` |
| `list` | `module(list)` — used for nested `Parallel` calls |

## Parallel vs Module.batch()

| | `dspy.Parallel` | `module.batch()` |
|---|---|---|
| Multiple different modules | ✅ | ❌ (single module only) |
| Nested fan-outs | ✅ | ❌ |
| Simplest API for one module | Use `module.batch()` instead | ✅ |
| Error collection | `return_failed_examples=True` | `return_failed_examples=True` |
| Straggler resubmission | `timeout` / `straggler_limit` | `timeout` / `straggler_limit` |
