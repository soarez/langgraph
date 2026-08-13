"""The payload encoder. Its contract is that it never raises."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from google.protobuf import json_format, struct_pb2
from pydantic import BaseModel

from langgraph.a2a._encoding import BYTES_KEY, decode, encode


class Colour(Enum):
    RED = "red"


class Model(BaseModel):
    name: str
    scores: list[int]


@dataclass
class Point:
    x: int
    y: int


class Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


def struct_round_trip(value: object) -> object:
    """Prove the encoded form is something a `data` part can actually carry."""
    proto = struct_pb2.Value()
    json_format.ParseDict(encode(value), proto)
    return json_format.MessageToDict(proto)


def test_scalars_pass_through() -> None:
    assert encode("x") == "x"
    assert encode(3) == 3
    assert encode(True) is True
    assert encode(None) is None


def test_non_finite_floats_become_strings() -> None:
    assert encode(math.inf) == "inf"
    assert encode(math.nan) == "nan"
    assert struct_round_trip({"n": math.inf}) == {"n": "inf"}


def test_bytes_survive_a_round_trip() -> None:
    encoded = encode(b"\x00\x01binary")

    assert set(encoded) == {BYTES_KEY}
    assert decode(struct_round_trip(b"\x00\x01binary")) == b"\x00\x01binary"


def test_pydantic_models_are_dumped() -> None:
    assert encode(Model(name="a", scores=[1, 2])) == {"name": "a", "scores": [1, 2]}


def test_dataclasses_are_dumped() -> None:
    assert encode(Point(1, 2)) == {"x": 1, "y": 2}


def test_enums_become_their_value() -> None:
    assert encode(Colour.RED) == "red"


def test_nested_structures_are_encoded_recursively() -> None:
    encoded = encode({"points": [Point(1, 2)], "model": Model(name="n", scores=[])})

    assert encoded == {
        "points": [{"x": 1, "y": 2}],
        "model": {"name": "n", "scores": []},
    }


def test_non_string_keys_are_coerced() -> None:
    assert encode({1: "a"}) == {"1": "a"}


def test_an_unencodable_object_becomes_its_repr() -> None:
    assert encode(Opaque()) == "<opaque>"
    assert struct_round_trip({"o": Opaque()}) == {"o": "<opaque>"}


def test_a_cycle_does_not_hang() -> None:
    cycle: dict = {}
    cycle["self"] = cycle

    assert encode(cycle) == {"self": "<circular>"}


def test_a_model_that_will_not_dump_still_encodes() -> None:
    class Exploding:
        def model_dump(self, **_kwargs: object) -> dict:
            raise RuntimeError("no")

        def __repr__(self) -> str:
            return "<exploding>"

    assert encode(Exploding()) == "<exploding>"
