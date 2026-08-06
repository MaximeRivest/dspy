---
title: "Representation Of A DSPy Program"
---
# The problem and the design

## What a DSPy program is

- A DSPy program is something you can call with typed (predefined) inputs and receive outputs. All that is needed to run and evaluate and program is in it's input / output pair.
  - So: it's a computer program, a function but not an 'app'. They are single turn, you should be able to uninstall (fully) a dspy program and get same results (performance) between each run. All that to say, its '1 user turn'. It's stateless.

### Reminder of what is a DSPy program, in more details

- A DSPy program is a small typed pipeline. Example: "question + documents → cited answer."
- The developer declares the interface: typed fields in, typed fields out.
- DSPy writes the prompts, parses the model replies, and optimizes the program. The optimizer learns better instructions and selects examples.
- The learned state includes instructions, demos, and possibly tuned weights.
- Upon execution, a dspy program resolve information from some ambient information (dspy.settings). Including the model, the prompt format, the tools, the model configs, the interpreter, and the credentials.

We generally say that a dspy program is optimized as one whole, because the prompt, demos and weights might have been optimized in such the way that they are because of anything in the program resolution path. Yet, we don't ship the program as a whole nor do we represent it in dspy. We should and can fix that. Here I argue we can and should fix that.

# Current DSPy program representation
![paste-2026-08-06T01-31-00](_assets/paste-2026-08-06T01-31-00.png)
Currently, in dspy, there is no `program`. 

`haiku_generator` is not a program, despite what our website says. That `program` object does not exist. 

A dspy `program` comes to act like one with these 3 things:

1. An instance (`module`). An in-memory object holding partial configuration: lists of demo examples, instruction strings, a config dict, and — this is important — an lm slot that is usually None, which literally means "decide later." The instance also has a `forward()` method — that's what runs when you call `haiku_generator(...)`. In Python everything is an object, so this code does exist "somewhere": as bytecode on the class, and as text in a .py file. But nothing in dspy ever reads it. When optimizers need the program's structure, dspy walks the instance's attributes and collects whatever sub-modules the developer happened to assign in `__init__` — a flat list, recovered by naming convention, not by analyzing the code. The control flow — the loops, the branches, what actually calls what — is invisible to the entire system.
2. Settings. the model, the prompt formatter, retry policy. Which values you get depends on when you ask and from where in the call stack you ask. This is not part of any program; it's shared mutable state that every program on the machine reads.
3. The dspy library's code. The prompt formats, parse rules, fallback logic, model-specific handling — behavior, as code paths. Not referenced by the instance, not recorded by the settings, selected implicitly by what pip install happened to put on the machine.

Currently, in dspy, the program only comes into existence during a call. When you invoke the `module` instance, at each internal prediction step, dspy assembles the actual program on the fly: it reads the instance's parameter bag, resolves the holes against whatever the ambient settings contain at that instant, and runs the resolved values through whatever logic this version of the library has. That assembled thing — the real program, the one whose behavior you measure — exists smeared across stack frames for the duration of the call, and is destroyed on return. It is never reified. There is no moment at which you can point at it, hash it, diff it, or store it.

So, each call is assembling a program 'live' from three uncoordinated sources. If the user or its environment change between to program run: a change in a setting between them, entering a context manager, upgrading the dspy library; the performance of the program may/will change.

Thus, "saving" a program is impossible (currently). We cannot simply serialize a program. There is no program! Save, currently write down the module instance (1). We have loaders that attempt to re-perform the assembly but it does it partially, and against it's own versions of (2) and (3). Each customization in lms, adapters, tools, interpreters has to try to figure out for itself of to make a program containing it, shippable. A second consequence of the lack of a program representation is that we have no concept of 'identity', "did the program change" cannot be answered, we don't have 2 entities to compare.

This is a system with no linked artifact and no link step. We have only source fragments, a dynamically-scoped environment, and an interpreter that re-resolves every symbol at every call site, on every call.

## Observed failures

- When a user builds a program with formatter B and saves it, the file does not record formatter B. When a different environment loads the file, dspy binds default formatter A. dspy shows no error. The performance changes, the instruction to the model changes.
- When a user saves a program, dspy does not write the generation settings. The loaded program runs with default settings. Temperature, native vs not native tool calls, etc.
- When a user saves a program with tools, dspy writes only the tool names. When a different user loads the file, dspy binds that user's functions to the names. dspy does not compare the functions. The user of the program has the burden of populating its python scope with the right tool and tool definition. ReAct and RLMs would run fine until they reach that undefined tool and crash (are run the wrong tool that happens to have the name).
- When a user loads a file that parses, the load operation succeeds. dspy does not verify that the loaded program has the same behavior. The errors appear later as quality decrease.

> High level stuff below, summarized form a much bigger spec on my dspy fork.

## The design of a Program representation

- The system shall represent a program as one serialized, typed artifact: the ProgramIR.
- The ProgramIR shall contain the module tree, the typed interfaces, the learned state, the resolved settings, the prompt-format decisions as data, and the control flow as a restricted typed AST with a small closed grammar.
- The ProgramIR shall contain each tool as source code plus a schema that the loader verifies, not as a name.
- The system shall declare each shared component (model, formatter, tool) one time, in a named pool. Each call site shall bind a pool entry by name. This gives a symbol table plus use sites. The load operation performs the link step.
- If a binding does not resolve, then the loader shall refuse the load and shall name the missing entry. The loader shall not read the global settings.
- The system shall put each component in one of three tiers:
  - Bake: the artifact carries the component (learned state, settings, tool source, model weights when possible).
  - Declare: the artifact names a required capability. When the loader binds a model endpoint, the loader shall verify that the endpoint serves the declared weights before first use (when possible).
  - Credential: the artifact contains credential names only. The system shall not write credential values into the artifact.
- The system shall keep the typed interface frozen. The system shall record each implementation choice (prompt format, prompting strategy, model) as data. Because the choices are data, the optimizer can search them.
- The system shall make each optimizer checkpoint a complete ProgramIR. The checkpoint operation and the save operation shall use one code path. Content-addressed storage keeps the cost of many checkpoints small.
- When a new machine loads an artifact, the system shall do one of two things: give a program that shows equal behavior (same prompts, same parses, same traces on a fixed test set), or refuse with a message that names the missing part. A machine-checked corpus verifies the representation at the byte level. This follows the TASTy pattern: when the typed tree is the artifact, tools become possible.

## Helps the platform

- Deploy: the user copies the artifact and runs the link step. Today, no deployable artifact exists.
- Serve: an engine executes the ProgramIR. The engine schedules over program-as-data, as vLLM schedules over weights-as-data: it batches the calls, caches the prompt plans, and streams typed events. A prototype interpreter (about 60 lines over the closed grammar) shows trace equality with native execution.
- Optimize: an optimization run produces a sequence of (checkpoint, labeled change, metric change). Each candidate is a complete artifact. The user can compare, revert, and attribute each candidate.
- Observe: the system annotates the same tree with run data: cost per node and metric quality per node. The system separates the declared outputs from the debug data.

Summary: deploy, serve, optimize, and observe are four views of one serialized object. A live object graph with global settings cannot support these four views. The ProgramIR is the substrate of the product.