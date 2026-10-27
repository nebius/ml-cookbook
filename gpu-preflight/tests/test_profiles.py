from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from gpu_preflight.compatibility import evaluate_cuda_target
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
                # The Nebius RTX PRO 6000 presets list no InfiniBand fabric, so
                # the profile declares no compute adapter.
                "rtx6000": None,
            },
        )


class RtxProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = load_profiles()["rtx6000"]

    def test_capability_is_sm120(self) -> None:
        self.assertEqual(self.profile.compute_capability_number, 120)

    def test_datacenter_blackwell_cubins_do_not_run_here(self) -> None:
        # SM100 and SM120 are both "Blackwell" but are separate targets.
        self.assertTrue(evaluate_cuda_target("sm_120", 120).compatible)
        self.assertFalse(evaluate_cuda_target("sm_100", 120).compatible)

    def test_family_target_covers_sm121(self) -> None:
        self.assertTrue(evaluate_cuda_target("sm_120f", 121).compatible)


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


class VersionFloorValidationTests(unittest.TestCase):
    """Version floors are compared component-wise against versions found in an image.

    A floor that is not dotted-numeric would raise deep inside a scan, so loading
    must reject it and name the file and field.
    """

    def test_dotted_numeric_floors_are_accepted(self) -> None:
        for field, value in (("minimum_cuda_version", "12.9"), ("minimum_nccl_version", "2.27.5")):
            with self.subTest(field=field):
                profile = _profile(MINIMAL_PROFILE + f'{field} = "{value}"\n')
                self.assertEqual(getattr(profile, field), value)
        for field, value in (("minimum_ucx_version", "1.18.0"), ("minimum_hpcx_version", "2.21.1")):
            with self.subTest(field=field):
                fabric = _fabric(MINIMAL_FABRIC + f'\n[validation]\n{field} = "{value}"\n')
                self.assertEqual(getattr(fabric, field), value)

    def test_absent_floors_stay_unset(self) -> None:
        profile = _profile(MINIMAL_PROFILE)
        self.assertIsNone(profile.minimum_cuda_version)
        self.assertIsNone(profile.minimum_nccl_version)

    MALFORMED = ("12", "latest", "v12.9", "12.x", "", "12.9.", "12..9")

    def test_malformed_profile_floors_are_rejected_at_load(self) -> None:
        for field in ("minimum_cuda_version", "minimum_nccl_version"):
            for value in self.MALFORMED:
                with self.subTest(field=field, value=value):
                    document = MINIMAL_PROFILE + f'{field} = "{value}"\n'
                    with self.assertRaisesRegex(ValueError, f"validation.{field}"):
                        _profile(document)

    def test_malformed_fabric_floors_are_rejected_at_load(self) -> None:
        for field in ("minimum_ucx_version", "minimum_hpcx_version"):
            for value in self.MALFORMED:
                with self.subTest(field=field, value=value):
                    document = MINIMAL_FABRIC + f'\n[validation]\n{field} = "{value}"\n'
                    with self.assertRaisesRegex(ValueError, f"validation.{field}"):
                        _fabric(document)

    def test_a_malformed_floor_fails_the_whole_directory_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "bad.toml").write_text(MINIMAL_PROFILE + 'minimum_cuda_version = "12"\n')
            with self.assertRaisesRegex(ValueError, "dotted numeric version"):
                load_profiles(path)

    def test_every_shipped_profile_and_fabric_floor_is_comparable(self) -> None:
        from gpu_preflight.inspection import _version_tuple

        for profile in load_profiles().values():
            for value in (profile.minimum_cuda_version, profile.minimum_nccl_version):
                if value is not None:
                    self.assertIsInstance(_version_tuple(value), tuple)
        for fabric in load_fabrics().values():
            for value in (fabric.minimum_ucx_version, fabric.minimum_hpcx_version):
                if value is not None:
                    self.assertIsInstance(_version_tuple(value), tuple)


if __name__ == "__main__":
    unittest.main()
