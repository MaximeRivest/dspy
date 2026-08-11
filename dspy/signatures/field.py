import warnings

import pydantic

# Marks a signature field whose type the user never declared (defaulted to str).
IS_TYPE_UNDEFINED = "IS_TYPE_UNDEFINED"

# The following arguments can be used in DSPy InputField and OutputField in addition
# to the standard pydantic.Field arguments. We just hope pydanitc doesn't add these,
# as it would give a name clash.
DSPY_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "role", "__dspy_field_type", IS_TYPE_UNDEFINED]

# The closed vocabulary of semantic roles a field may declare: the field's
# relationship to the LM exchange, separate from its data shape. Roles are
# signature-level intent; the inference strategy answering a role is an
# adapter/engine concern. See roadmap/epic-C-semantic-roles.md.
SEMANTIC_ROLES = frozenset(
    ["plain", "reasoning", "tools", "tool_calls", "citations", "history", "media", "code"]
)

# Version of the role vocabulary; carried in every serialized preset's
# versions block (D-024). Extending SEMANTIC_ROLES is a versioned act.
SEMANTIC_ROLES_VERSION = "1.0.0"

_DEPRECATED_FIELD_ARGS = {
    "prefix": (
        "The 'prefix' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
    "format": (
        "The 'format' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
    "parser": (
        "The 'parser' argument in InputField/OutputField is deprecated and has no effect in DSPy. "
        "It will be removed in a future version."
    ),
}

PYDANTIC_CONSTRAINT_MAP = {
    "gt": "greater than: ",
    "ge": "greater than or equal to: ",
    "lt": "less than: ",
    "le": "less than or equal to: ",
    "min_length": "minimum length: ",
    "max_length": "maximum length: ",
    "multiple_of": "a multiple of the given number: ",
    "allow_inf_nan": "allow 'inf', '-inf', 'nan' values: ",
}


def move_kwargs(**kwargs):
    # Pydantic doesn't allow arbitrary arguments to be given to fields,
    # but asks that
    # > any extra data you want to add to the JSON schema should be passed
    # > as a dictionary to the json_schema_extra keyword argument.
    # See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield
    pydantic_kwargs = {}
    json_schema_extra = {}
    for k, v in kwargs.items():
        if k in DSPY_FIELD_ARG_NAMES:
            json_schema_extra[k] = v
        else:
            pydantic_kwargs[k] = v
    # Also copy over the pydantic "description" if no dspy "desc" is given.
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]
    # An explicit semantic role is validated eagerly and stored under
    # "semantic_role" ("role" alone would collide with the render-field
    # input/output direction key).
    if "role" in json_schema_extra:
        role = json_schema_extra.pop("role")
        if role not in SEMANTIC_ROLES:
            raise ValueError(
                f"Unknown semantic role {role!r}. Valid roles: {sorted(SEMANTIC_ROLES)}."
            )
        json_schema_extra["semantic_role"] = role
    constraints = _translate_pydantic_field_constraints(**kwargs)
    if constraints:
        json_schema_extra["constraints"] = constraints
    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def _translate_pydantic_field_constraints(**kwargs):
    """Extracts Pydantic constraints and translates them into human-readable format."""

    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")

    return ", ".join(constraints)


def _warn_deprecated_field_args(**kwargs):
    for arg, message in _DEPRECATED_FIELD_ARGS.items():
        if arg in kwargs:
            warnings.warn(message, DeprecationWarning, stacklevel=3)


def InputField(**kwargs): # noqa: N802
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs): # noqa: N802
    _warn_deprecated_field_args(**kwargs)
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))
