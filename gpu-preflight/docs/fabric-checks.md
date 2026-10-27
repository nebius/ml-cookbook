# Compute fabric, UCX, and HPC-X checks

How GPU Preflight derives the fabric support floor for a hardware target, and what it
reports about a UCX or HPC-X installation found in an image.

Back to the [GPU Preflight README](../README.md).

## Compute fabric profiles

Each hardware profile names the compute-fabric adapter of its Nebius platform under `[fabric]`, and the adapters themselves live in `src/gpu_preflight/data/fabrics/`. Adapter support is a property of the NIC generation rather than the GPU, so one adapter entry and one citation cover every platform that ships it: ConnectX-7 for H100, H200, and B200; ConnectX-8 for B300 and GB300; ConnectX-9 for Vera Rubin.

An adapter entry records `minimum_hpcx_version`, the first HPC-X release whose change log adds that adapter, and `minimum_ucx_version`, the UCX release bundled with it. The UCX floor is therefore derived rather than separately published: NVIDIA states which HPC-X release supports an adapter, not a standalone UCX minimum, and each entry's notes say so. ConnectX-9 carries no floor, because no HPC-X release has announced support for it.

## UCX and HPC-X checks

UCX is optional. NCCL reaches InfiniBand through its own plugin, so an image with no UCX produces no finding at all. When an image does ship UCX or HPC-X, GPU Preflight reports:

- `GPF-UCX-001` / `GPF-UCX-006`: the installed releases and transport-module directories, or that the release could not be established. A bare `libucp.so.0` soname identifies the ABI, not the UCX release, so an installation without package metadata, an `HPCX_VERSION`-style environment variable, or an `/opt/hpcx/VERSION` file is reported as undetermined rather than guessed.
- `GPF-UCX-002`: a modular UCX build whose transport modules exclude CUDA. Such a build loads and passes connectivity tests while staging every GPU buffer through host memory. The finding requires a module directory that already holds `libuct_*` objects, because a UCX built with `--enable-modules=no` links CUDA support into `libuct` itself and legitimately ships no separate module.
- `GPF-UCX-003`: NIXL installed with no UCX runtime at all. UCX is NIXL's default transfer backend.
- `GPF-UCX-004` / `GPF-UCX-005`: detected releases against the adapter floor described above.

As everywhere else in the report, these are static observations. GPU Preflight cannot establish which UCX installation a workload loads, nor which transport it selects at runtime; `ucx-transport` is in each profile's required hardware checks for that reason.
