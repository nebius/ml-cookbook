from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from gpu_preflight.profiles import (
    load_fabrics,
    load_profiles,
    parse_fabric,
    parse_profile,
)

MINIMAL_PROFILE = """
schema_version = 1
id = "test-gpu"
aliases = []
display_name = "Test GPU"
maturity = "supported"

[hardware]
cpu_architectures = ["amd64"]
gpu_architecture = "test"
compute_capability = "9.0"
native_sm_targets = ["sm_90"]
nebius_platforms = []

[validation]
runtime_checks = []
"""

MINIMAL_FABRIC = """
schema_version = 1
id = "test-nic"
display_name = "Test NIC"
generation = 1
"""


def _profile(document: str):
    return parse_profile(tomllib.loads(document), "test.toml")


def _fabric(document: str):
    return parse_fabric(tomllib.loads(document), "test.toml")


class FabricTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fabrics = load_fabrics()
        cls.profiles = load_profiles()

    def test_built_in_adapters_are_loaded(self) -> None:
        self.assertEqual(
            {identifier: fabric.generation for identifier, fabric in self.fabrics.items()},
            {"connectx-7": 7, "connectx-8": 8, "connectx-9": 9},
        )

    def test_every_adapter_cites_a_source(self) -> None:
        for identifier, fabric in self.fabrics.items():
            with self.subTest(fabric=identifier):
                self.assertTrue(fabric.sources, "an adapter must cite its evidence")

    def test_version_floors_are_optional(self) -> None:
        # Floors stay unset until an NVIDIA adapter-support citation fixes them, so
        # the loader must treat an absent [validation] table as valid rather than empty.
        for identifier, fabric in self.fabrics.items():
            with self.subTest(fabric=identifier):
                for floor in (fabric.minimum_ucx_version, fabric.minimum_hpcx_version):
                    self.assertIsInstance(floor, (str, type(None)))

    def test_profiles_map_to_known_adapters(self) -> None:
        self.assertEqual(
            {identifier: profile.compute_fabric for identifier, profile in self.profiles.items()},
            {
                "h100": "connectx-7",
                "h200": "connectx-7",
                "b200": "connectx-7",
                "b300": "connectx-8",
                "gb300": "connectx-8",
                "vr": "connectx-9",
            },
        )


class FabricParsingTests(unittest.TestCase):
    def test_minimal_fabric_parses(self) -> None:
        fabric = _fabric(MINIMAL_FABRIC)
        self.assertEqual(fabric.id, "test-nic")
        self.assertIsNone(fabric.minimum_ucx_version)
        self.assertIsNone(fabric.minimum_hpcx_version)

    def test_floors_parse_when_present(self) -> None:
        fabric = _fabric(
            MINIMAL_FABRIC
            + '\n[validation]\nminimum_ucx_version = "1.17"\nminimum_hpcx_version = "2.21"\n'
        )
        self.assertEqual(fabric.minimum_ucx_version, "1.17")
        self.assertEqual(fabric.minimum_hpcx_version, "2.21")

    def test_generation_must_be_a_positive_integer(self) -> None:
        for value in ('"8"', "0", "true"):
            with self.subTest(generation=value):
                document = MINIMAL_FABRIC.replace("generation = 1", f"generation = {value}")
                with self.assertRaisesRegex(ValueError, "generation"):
                    _fabric(document)

    def test_schema_version_must_be_one(self) -> None:
        document = MINIMAL_FABRIC.replace("schema_version = 1", "schema_version = 2")
        with self.assertRaisesRegex(ValueError, "schema_version"):
            _fabric(document)


class ProfileFabricTests(unittest.TestCase):
    def test_fabric_table_is_optional(self) -> None:
        self.assertIsNone(_profile(MINIMAL_PROFILE).compute_fabric)

    def test_adapter_is_normalized(self) -> None:
        document = MINIMAL_PROFILE + '\n[fabric]\ncompute_adapter = "ConnectX-8"\n'
        self.assertEqual(_profile(document).compute_fabric, "connectx-8")

    def test_fabric_must_be_a_table(self) -> None:
        document = MINIMAL_PROFILE.replace("[hardware]", 'fabric = "connectx-8"\n\n[hardware]')
        with self.assertRaisesRegex(ValueError, "fabric must be a table"):
            _profile(document)

    def test_unknown_adapter_is_rejected(self) -> None:
        document = MINIMAL_PROFILE + '\n[fabric]\ncompute_adapter = "connectx-99"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test-gpu.toml").write_text(document)
            with self.assertRaisesRegex(ValueError, "unknown compute fabric"):
                load_profiles(path)


if __name__ == "__main__":
    unittest.main()
