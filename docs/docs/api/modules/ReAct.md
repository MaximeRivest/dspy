# dspy.ReAct

`dspy.ReAct` implements the *Reasoning and Acting* paradigm for building tool-using agents. The model iteratively reasons about the current situation, calls tools to gather information, and finally produces the output fields. It works with any DSPy signature thanks to signature polymorphism.

<!-- START_API_REF -->
::: dspy.ReAct
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
            - truncate_trajectory
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

### Basic tool-using agent

```python
import dspy

dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."

react = dspy.ReAct("question -> answer", tools=[get_weather])
result = react(question="What is the weather in Tokyo?")
print(result.answer)
```

### Multiple tools

```python
def search(query: str) -> str:
    """Search the web for information."""
    return f"Result for '{query}': Python was created by Guido van Rossum."

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

react = dspy.ReAct("question -> answer", tools=[search, calculator])
result = react(question="Who created Python?")
print(result.answer)
```

### Accessing the trajectory

The result includes the full trajectory of thoughts, tool calls, and observations:

```python
result = react(question="What is the weather in Paris?")
for key, value in result.trajectory.items():
    print(f"{key}: {value}")
```

### How ReAct works

1. **Think** — The model produces a thought about the current situation
2. **Act** — It selects a tool and provides arguments
3. **Observe** — The tool is executed and the result is appended to the trajectory
4. **Repeat** until the model calls the built-in `finish` tool or `max_iters` is reached
5. **Extract** — A `ChainOfThought` fallback extracts the final output fields
