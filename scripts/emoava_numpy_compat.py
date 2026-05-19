from __future__ import annotations


def patch_numpy_legacy_aliases() -> None:
    """Restore NumPy 1.x aliases that older DECA/chumpy code still imports.

    NumPy 2.x removed aliases such as np.int / np.float / np.object. The
    EmoAva / DECA stack still relies on these legacy names through chumpy.
    """

    import numpy as np

    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "long": int,
    }
    for name, value in legacy_aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)
