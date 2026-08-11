# 09 media-shapes — the two-layer rule made concrete

## What this shows

The signature says `PIL.Image.Image`; the entry says
`(shape: image, wire: base64/png)` plus a `frontend_bindings.python`
annotation. This is the north star's cross-language test verbatim:
"data-only" means the thing ITSELF crosses languages, so a codec is
data only when expressed against the neutral shape vocabulary, and
the host type is a per-frontend binding, NEVER IR content. The
`media` strategy (`native_parts`) then decides the exchange: the
field's slot emits an ImagePart at its template position — the
part-emission rule adapter-ir-spec.md section 2 already pins.

Receivers ignore bindings for frontends they are not: a Go receiver
reads shape+wire, materializes `image.Image`, and never sees Python.
The binding exists so a Python REload restores the exact promise the
signature made (the Arrow/protobuf i64-vs-BigInt pattern).

## Data ladder placement

- The shape codec: data, against TWO versioned vocabularies —
  `shapes` (what an image IS, pinned decode semantics) and the wire
  encodings. No trust question anywhere.

## What today's dspy does instead

`dspy.Image` is a dspy-owned wrapper type, not a neutral shape; the
content-splitting `{image}` mechanism exists in the reference
implementation but the shape/wire/binding triple is not serialized —
a dumped entry cannot say "this field is an image on the wire".

## PROPOSED spellings

- The shape-codec object: `kind: "shape"`, `shape`, `wire`
  (`encoding` + `media_type`), `frontend_bindings` ({frontend ->
  host type name}).
- `versions.shapes` — the Epic B shapes vocabulary versioned in the
  entry.
- `dump_entry(for_signature=...)` — per-field codecs only exist
  relative to a signature; whether per_field entries belong in the
  ADAPTER entry or in the ProgramIR's per-predictor binding is a real
  open question (the adapter is signature-independent by definition).
