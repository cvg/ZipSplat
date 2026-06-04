"""Type definitions and global variables.

Adapted from gluefactory (https://github.com/cvg/glue-factory), Apache-2.0.
"""

from __future__ import annotations

from typing import Any, TypeAlias

STRING_CLASSES = (str, bytes)

INVALID_DEPTH = -1

Key: TypeAlias = str | tuple[str, ...]
Value: TypeAlias = Any
Tree: TypeAlias = dict[Key, Value]
