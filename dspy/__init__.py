"""dspy rebuilt on the IR foundation (branch greenfield-ir).

Stage A1 stub: signatures + roles only. Core, LM, adapters v2, the
execution spine, modules, and optimizers land in later stages.
"""

from dspy.signatures import (
    InputField,
    OutputField,
    Signature,
    SignatureMeta,
    ensure_signature,
    make_signature,
)
from dspy import roles

from dspy.__metadata__ import __author__, __author_email__, __description__, __name__, __url__, __version__
