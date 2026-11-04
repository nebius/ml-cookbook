from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from gpu_preflight.compatibility import evaluate_cuda_target
from gpu_preflight.facts import normalize_package_name, package_versions
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


class RuntimePackageFloorTests(unittest.TestCase):
    """Runtime compilers, DSLs and kernel libraries emit device code when the workload
    runs, so an image holds no artifact to inspect for them and the installed version
    is the only static evidence. A floor is recorded only where a release note states
    one, which is why the field is optional.
    """

    ENTRY = '\n[[validation.runtime_packages]]\npackage = "triton"\nrole = "compiler"\n'

    def test_a_floor_and_its_citation_are_parsed(self) -> None:
        profile = _profile(
            MINIMAL_PROFILE
            + self.ENTRY
            + 'minimum_version = "3.5.1"\n'
            + 'source = "https://github.com/triton-lang/triton/releases"\n'
        )
        (floor,) = profile.runtime_packages
        self.assertEqual(floor.package, "triton")
        self.assertEqual(floor.role, "compiler")
        self.assertEqual(floor.minimum_version, "3.5.1")
        self.assertEqual(floor.source, "https://github.com/triton-lang/triton/releases")

    def test_a_package_may_be_listed_without_a_floor(self) -> None:
        # Listing it without a floor is the point: the package is reported as
        # present-but-unassessed rather than silently assumed compatible.
        profile = _profile(MINIMAL_PROFILE + self.ENTRY)
        (floor,) = profile.runtime_packages
        self.assertIsNone(floor.minimum_version)

    def test_profiles_without_the_table_keep_an_empty_tuple(self) -> None:
        self.assertEqual(_profile(MINIMAL_PROFILE).runtime_packages, ())

    def test_a_malformed_floor_is_rejected_at_load(self) -> None:
        for value in VersionFloorValidationTests.MALFORMED:
            with self.subTest(value=value):
                document = MINIMAL_PROFILE + self.ENTRY + f'minimum_version = "{value}"\n'
                with self.assertRaisesRegex(
                    ValueError, "validation.runtime_packages.minimum_version"
                ):
                    _profile(document)

    def test_an_unknown_role_is_rejected_at_load(self) -> None:
        document = MINIMAL_PROFILE + (
            '\n[[validation.runtime_packages]]\npackage = "triton"\nrole = "transpiler"\n'
        )
        with self.assertRaisesRegex(ValueError, "validation.runtime_packages.role"):
            _profile(document)

    def test_one_package_listed_twice_is_rejected(self) -> None:
        # Two entries for one package would make the floor depend on table order.
        with self.assertRaisesRegex(ValueError, "twice"):
            _profile(MINIMAL_PROFILE + self.ENTRY + self.ENTRY)

    def test_applicability_defaults_to_supported(self) -> None:
        (floor,) = _profile(MINIMAL_PROFILE + self.ENTRY).runtime_packages
        self.assertEqual(floor.applicability, "supported")

    def test_a_package_can_be_marked_not_applicable(self) -> None:
        # Hardware support does not always move forward along one version line, so a
        # profile has to be able to say "no release of this targets us".
        document = MINIMAL_PROFILE + self.ENTRY + 'applicability = "not-applicable"\n'
        (floor,) = _profile(document).runtime_packages
        self.assertEqual(floor.applicability, "not-applicable")

    def test_an_unknown_applicability_is_rejected_at_load(self) -> None:
        document = MINIMAL_PROFILE + self.ENTRY + 'applicability = "maybe"\n'
        with self.assertRaisesRegex(ValueError, "validation.runtime_packages.applicability"):
            _profile(document)

    def test_a_floor_on_a_not_applicable_package_is_rejected(self) -> None:
        # A floor says every later release works here; the marker says none does.
        document = (
            MINIMAL_PROFILE
            + self.ENTRY
            + 'applicability = "not-applicable"\n'
            + 'minimum_version = "3.5.1"\n'
        )
        with self.assertRaisesRegex(ValueError, "cannot set minimum_version"):
            _profile(document)

    def test_only_packages_artifact_inspection_cannot_see_are_listed(self) -> None:
        # cuobjdump reads a compiled extension's sm_ targets from the artifact and
        # names the file, so listing those here would be a hand-maintained roster of
        # libraries that artifact inspection already discovers. Only JIT compilers,
        # DSLs and host-side libraries belong.
        compiled = {"flash-attn", "flash-attn-3", "nixl", "xformers", "transformer-engine"}
        for name, profile in load_profiles().items():
            with self.subTest(profile=name):
                listed = {floor.package for floor in profile.runtime_packages}
                self.assertEqual(listed & compiled, set())

    def test_every_profile_assesses_the_same_packages(self) -> None:
        # The package table was built for Blackwell first, which left Hopper, RTX PRO
        # 6000 and Vera Rubin silently skipping packages the Blackwell reports judged.
        # Silence reads as "nothing to say" rather than "never looked", so the set is
        # held identical across profiles; what differs is the floor, not the coverage.
        profiles = load_profiles()
        expected = {floor.package for floor in profiles["b300"].runtime_packages}
        for name, profile in profiles.items():
            with self.subTest(profile=name):
                self.assertEqual({f.package for f in profile.runtime_packages}, expected)

    def test_cute_dsl_floors_track_the_architecture(self) -> None:
        # Each is the earliest release whose notes name that architecture.
        floors = {
            name: next(
                f.minimum_version
                for f in profile.runtime_packages
                if f.package == "nvidia-cutlass-dsl"
            )
            for name, profile in load_profiles().items()
        }
        self.assertEqual(floors["h100"], "4.0.0")
        self.assertEqual(floors["b200"], "4.0.0")
        self.assertEqual(floors["b300"], "4.3.1")
        self.assertEqual(floors["gb300"], "4.4.0")
        self.assertEqual(floors["rtx6000"], "4.5.0")
        self.assertEqual(floors["vr"], "4.8.0")

    def test_package_names_match_what_images_actually_install(self) -> None:
        # The floor lookup is an exact match on the normalized distribution name, so a
        # profile naming "flashinfer" or "flash-attn-4" wrongly assesses nothing at all.
        # These are the names seen in real image inventories.
        installed = package_versions(
            (
                "flashinfer-python==0.7.2",
                "flashinfer-cubin==0.7.2",
                "flashinfer-jit-cache-sm103a==0.7.2",
                "flash-attn-4==4.0.0b37",
                "flash_attn==2.8.3",
                "nixl==1.5.2",
                "nixl-cu13==1.5.2",
                "nvidia-nvshmem-cu12==3.4.5",
                "nvidia-nvshmem-cu13==3.4.5",
                "libnvshmem3-cuda-12==3.2.5-1",
                "libnvshmem3-cuda-13==3.7.2-1",
                "nvidia-cutlass-dsl==4.8.0",
                "triton==3.8.0",
            )
        )
        for name, profile in load_profiles().items():
            for floor in profile.runtime_packages:
                if floor.package in {"flash-attn-3", "liger-kernel", "numba"}:
                    continue
                for alias in floor.lookup_names:
                    with self.subTest(profile=name, package=alias):
                        self.assertIn(normalize_package_name(alias), installed)

    def test_rubin_floors_come_from_release_notes(self) -> None:
        # Triton 3.8.0 is the release that announces SM107; nothing earlier names it.
        floors = {f.package: f for f in load_profiles()["vr"].runtime_packages}
        self.assertEqual(floors["triton"].minimum_version, "3.8.0")
        self.assertTrue(floors["triton"].source)

    def test_packages_without_a_published_floor_stay_unfloored(self) -> None:
        # numba-cuda announces no architectures at all and no Triton release note
        # names sm_120, so neither may carry an invented floor.
        profiles = load_profiles()
        for name in profiles:
            floors = {f.package: f for f in profiles[name].runtime_packages}
            with self.subTest(profile=name):
                self.assertIsNone(floors["numba"].minimum_version)
        rtx = {f.package: f for f in profiles["rtx6000"].runtime_packages}
        self.assertIsNone(rtx["triton"].minimum_version)

    def test_shipped_blackwell_profiles_cite_every_floor(self) -> None:
        # A floor without a citation is a number nobody can check, and these were
        # read out of vendor release notes one at a time.
        floors = 0
        for profile in load_profiles().values():
            for floor in profile.runtime_packages:
                if floor.minimum_version is not None:
                    floors += 1
                    self.assertTrue(floor.source, f"{profile.id}/{floor.package}")
        self.assertGreater(floors, 0)

    def test_gb300_needs_a_newer_cute_dsl_than_b300(self) -> None:
        # SM103 reached CuTe DSL in CUTLASS 4.3.1, but the GB300 platform only in
        # 4.4.0. Sharing an SM target does not mean sharing a floor.
        profiles = load_profiles()
        floors = {
            name: {
                floor.package: floor.minimum_version for floor in profiles[name].runtime_packages
            }
            for name in ("b300", "gb300")
        }
        self.assertEqual(floors["b300"]["nvidia-cutlass-dsl"], "4.3.1")
        self.assertEqual(floors["gb300"]["nvidia-cutlass-dsl"], "4.4.0")
