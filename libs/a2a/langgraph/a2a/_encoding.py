"""Turning arbitrary Python into something an A2A `data` part can carry.

`interrupt()` accepts any Python object. A2A `data` parts carry
`google.protobuf.Struct`, which is JSON. The gap between the two is this
module's whole job, and it must never raise: an unserialisable interrupt
payload is a documentation problem for the graph author, not a protocol error
for the caller.

The rules, in order:

| Input | Output |
|---|---|
| `None`, `bool`, `int`, `float`, `str` | itself (non-finite floats become `str`) |
| `bytes` / `bytearray` | `{"__bytes__": "<base64>"}` |
| pydantic model (`model_dump`) | the dumped mapping, encoded recursively |
| dataclass instance | its fields, encoded recursively |
| `Mapping` | keys coerced to `str`, values encoded recursively |
| `Sequence` / `set` (not `str`/`bytes`) | a list, encoded recursively |
| `Enum` | its value, encoded |
| anything else | `repr()`, as a string |

Cycles are broken with the string `"<circular>"`.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import math
from collections.abc import Mapping, Sequence, Set
from typing import Any

BYTES_KEY = "__bytes__"
"""Key under which `bytes` survive a round-trip through a `data` part."""

_MAX_DEPTH = 32


def encode(value: Any) -> Any:
    """Encode `value` as JSON-compatible data. Never raises."""
    return _encode(value, depth=0, seen=frozenset())


def decode(value: Any) -> Any:
    """Reverse `encode` where it is reversible — currently the `bytes` wrapper."""
    if isinstance(value, Mapping):
        if set(value) == {BYTES_KEY} and isinstance(value[BYTES_KEY], str):
            try:
                return base64.b64decode(value[BYTES_KEY])
            except Exception:
                return value
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def _encode(value: Any, *, depth: int, seen: frozenset[int]) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        # Struct is JSON: NaN and the infinities have no representation.
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes | bytearray):
        return {BYTES_KEY: base64.b64encode(bytes(value)).decode()}
    if isinstance(value, enum.Enum):
        return _encode(value.value, depth=depth, seen=seen)
    if depth >= _MAX_DEPTH:
        return repr(value)
    if id(value) in seen:
        return "<circular>"

    seen = seen | {id(value)}
    depth += 1

    dumped = _pydantic_dump(value)
    if dumped is not None:
        return _encode(dumped, depth=depth, seen=seen)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode(
                getattr(value, field.name, None), depth=depth, seen=seen
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _encode(item, depth=depth, seen=seen)
            for key, item in value.items()
        }
    if isinstance(value, Set):
        return [_encode(item, depth=depth, seen=seen) for item in value]
    if isinstance(value, Sequence):
        return [_encode(item, depth=depth, seen=seen) for item in value]
    return repr(value)


def _pydantic_dump(value: Any) -> Any:
    """`value.model_dump(mode="json")` if `value` is a pydantic model, else `None`."""
    dump = getattr(value, "model_dump", None)
    if dump is None or isinstance(value, type):
        return None
    try:
        return dump(mode="json")
    except Exception:
        try:
            return dump()
        except Exception:
            return None
