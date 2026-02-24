# dspy.ProgramOfThought

`dspy.ProgramOfThought` prompts the language model to write and execute Python code to solve a task programmatically. If execution fails, the error is fed back to the model for automatic retry. Requires [Deno](https://docs.deno.com/runtime/getting_started/installation/) for sandboxed execution.

<!-- START_API_REF -->
::: dspy.ProgramOfThought
    handler: python
    options:
        members:
            - __call__
            - acall
            - batch
            - deepcopy
            - dump_state
            - forward
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
            - reset_copy
            - save
            - set_lm
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

### Basic code generation

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

pot = dspy.ProgramOfThought("question -> answer")
result = pot(question="What is 15 * 7?")
print(result.answer)
```

### How it works

1. **Generate** — `ChainOfThought` generates Python code to solve the task
2. **Execute** — The code runs in a sandboxed `PythonInterpreter` (Deno-backed)
3. **Retry on error** — If execution fails, the error message is fed back and new code is generated (up to `max_iters` times)
4. **Extract** — A final `ChainOfThought` call extracts the answer from the code output

!!! warning
    This module requires **Deno** to be installed for the sandboxed Python interpreter. See the [Deno installation guide](https://docs.deno.com/runtime/getting_started/installation/).
