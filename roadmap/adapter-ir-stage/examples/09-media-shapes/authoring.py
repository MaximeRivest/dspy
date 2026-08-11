# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import PIL.Image

import dspy


class Caption(dspy.Signature):
    """Describe the photo."""

    photo: PIL.Image.Image = dspy.InputField(role="media")
    caption: str = dspy.OutputField()


# The user writes the HOST type. That is the whole point: back-and-forth
# of host-language objects is a core power, never restricted. The
# two-layer rule does the rest at export:
#
#   PIL.Image.Image  --lowers to-->  (shape: image, wire: base64/png)
#                                    + a python frontend binding
#
# The IR carries only the neutral shape + wire encoding (Epic B shapes
# = the Adapter IR's type system, with pinned decode semantics). A TS
# receiver materializes a Blob; a Go receiver an image.Image; every
# frontend binds its native type. The binding is annotation, never IR
# content.
adapter = dspy.ChatAdapter()
program = dspy.Predict(Caption)

entry = adapter.dump_entry(for_signature=Caption)  # NEW: per-signature lowering
assert entry["codecs"]["per_field"]["photo"] == {
    "kind": "shape",
    "shape": "image",
    "wire": {"encoding": "base64", "media_type": "image/png"},
    "frontend_bindings": {"python": "PIL.Image.Image"},
}
