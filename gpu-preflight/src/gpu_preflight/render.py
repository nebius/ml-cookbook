"""Human-readable and machine-readable report rendering."""

from __future__ import annotations

import json
import re
from html import escape

from gpu_preflight.inspection import Finding, InspectionReport
from gpu_preflight.profiles import HardwareProfile

SEVERITY_LABEL = {
    "blocker": "BLOCKER",
    "warning": "WARNING",
    "info": "INFO",
}

SKIP_REASON_LABEL = {
    "no-cuobjdump": "CUDA analyzer unavailable",
    "scan-budget": "candidate-content budget",
    "artifact-limit": "artifact-size limit",
    "file-limit": "file-count limit",
}

NCCL_SOURCE_LABEL = {
    "image-config": "Image configuration",
    "system-package": "System package",
    "python-package": "Python package",
    "conda-package": "Conda package",
    "header": "Development header",
}

EXCEPTION_CLASSIFICATION_LABEL = {
    "likely-runtime-compatibility-gap": "Likely runtime compatibility gap",
    "optional-alternate-gpu-implementation": "Optional alternate-GPU implementation",
    "runtime-relevance-not-statically-established": (
        "Runtime relevance not statically established"
    ),
    "vendor-library-supported-by-release": "Supported by its CUDA release (informational)",
}

HARDWARE_CHECK_GUIDANCE = {
    "arm64-image": "Pull and start the resolved arm64 digest on the target host.",
    "cuda-runtime": "Run a framework GPU allocation and representative CUDA kernel.",
    "cuda-sm107-runtime": "Run a native SM107 kernel and verify it does not rely on an older cubin.",
    "nccl-intra-node": "Run NCCL collective tests across all GPUs in one node.",
    "nccl-mnnvl": "Run multi-node NVLink collective tests and confirm the intended MNNVL path.",
    "nccl-infiniband": "Run multi-node NCCL tests and confirm InfiniBand transport selection.",
    "gpu-direct-rdma": "Verify GPUDirect RDMA is selected and transfers complete without fallback.",
    "rack-fabric": "Run rack-scale collectives and verify the intended fabric topology.",
    "scaleout-network": "Run multi-node collectives over the production scale-out network.",
    "framework-smoke": "Import the framework, enumerate GPUs, and run a small tensor operation.",
    "workload-smoke": "Run a representative reduced workload and verify correctness and stability.",
}


def _compact(values: list[str] | tuple[str, ...], limit: int = 16) -> str:
    if not values:
        return "none detected"
    visible = ", ".join(values[:limit])
    return f"{visible}, plus {len(values) - limit} more" if len(values) > limit else visible


def _effective_values(observations: dict[str, object], base: str, image: str) -> list[str]:
    return sorted(
        {
            *[str(value) for value in observations.get(base, [])],
            *[str(value) for value in observations.get(image, [])],
        }
    )


def _cuda_release_versions(values: list[str]) -> list[str]:
    """Collapse package-build versions when a toolkit release label is also available."""
    selected: dict[tuple[int, int] | tuple[str, str], str] = {}
    for value in values:
        parts = tuple(int(part) for part in re.findall(r"\d+", value))
        family: tuple[int, int] | tuple[str, str]
        family = (parts[0], parts[1]) if len(parts) >= 2 else (value, value)
        existing = selected.get(family)
        if existing is None or len(parts) < len(tuple(re.findall(r"\d+", existing))):
            selected[family] = value
    return sorted(selected.values())


def _hardware_check_text(check: str) -> str:
    guidance = HARDWARE_CHECK_GUIDANCE.get(check, "Execute and retain target-hardware evidence.")
    return f"{check}: {guidance}"


def _artifact_coverage_text(
    percent: float | None,
    covered: int | None,
    eligible: int | None,
) -> str:
    if percent is None:
        return "unavailable"
    counts = f" ({covered}/{eligible} eligible runtime inspection units)" if eligible else ""
    return f"{percent:.1f}%{counts}"


def _nccl_evidence_lines(
    observations: dict[str, object], effective_versions: list[str]
) -> list[str]:
    evidence = observations.get("nccl_evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return []
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for item in evidence:
        if not isinstance(item, dict):
            continue
        version = str(item.get("version", "unknown"))
        source = str(item.get("source", "unknown"))
        path = str(item.get("path") or "")
        grouped.setdefault((version, source, path), set()).add(str(item.get("name", "unknown")))
    lines = ["  NCCL installations and version evidence:"]
    for version in effective_versions:
        lines.append(f"    - {version}:")
        for (item_version, source, path), names in sorted(grouped.items()):
            if item_version != version:
                continue
            location = f" ({path})" if path else ""
            lines.append(
                f"      - {NCCL_SOURCE_LABEL.get(source, source)}: "
                f"{', '.join(sorted(names))}{location}"
            )
    if len(effective_versions) > 1:
        lines.append(
            "    Effective loaded NCCL version: not statically established; multiple version "
            "sources were found. Confirm the loaded library during runtime validation."
        )
    else:
        lines.append(
            "    Effective loaded NCCL version: confirm during runtime validation; static "
            "inventory does not prove dynamic-library selection."
        )
    return lines


def _cuda_compatibility_exception_lines(report: InspectionReport) -> list[str]:
    exceptions = report.observations.get("cuda_compatibility_exceptions", [])
    groups = report.observations.get("cuda_compatibility_exception_groups", [])
    if not isinstance(exceptions, list) or not exceptions or not isinstance(groups, list):
        return []
    display_names = {target.target: target.display_name for target in report.targets}
    lines = [
        "",
        "CUDA compatibility review summary:",
        (
            "  Counts describe binary files and grouped dispatch catalogs, not operations, "
            "features, or distinct workload code paths. Presence in the image does not prove "
            "the workload loads an item."
        ),
    ]
    for target in report.targets:
        target_groups = [
            item
            for item in groups
            if isinstance(item, dict) and item.get("target") == target.target
        ]
        if not target_groups:
            continue
        lines.append(f"  {display_names[target.target]} ({target.target}):")
        for item in target_groups:
            classification = EXCEPTION_CLASSIFICATION_LABEL.get(
                str(item.get("classification")), str(item.get("classification"))
            )
            lines.extend(
                (
                    f"    - {item.get('component')}: {classification}",
                    f"      Reason: {item.get('reason_summary')}",
                    f"      Review units: {item.get('exception_count')}",
                    f"      Binary files represented: {item.get('file_count')}",
                    "      Representative examples:",
                )
            )
            lines.extend(f"        - {path}" for path in item.get("representative_paths", []))
    lines.extend(
        (
            "  Full inventory:",
            "    - JSON: open the sibling report.json file and inspect "
            "observations.cuda_compatibility_exceptions.",
            "    - HTML: open the sibling report.html file and expand the complete CUDA "
            "compatibility inventory.",
        )
    )
    return lines


def _rootfs_statistics_lines(
    observations: dict[str, object], settings: dict[str, object]
) -> list[str]:
    statistics = observations.get("rootfs_statistics", {})
    if not isinstance(statistics, dict):
        return []
    lines: list[str] = []
    for platform, value in sorted(statistics.items()):
        if not isinstance(value, dict):
            continue
        analyzed = int(value.get("cuobjdump_files_analyzed", 0))
        targets_found = int(value.get("cuobjdump_targets_found", 0))
        no_device_code = int(value.get("cuobjdump_no_device_code", 0))
        incomplete = int(value.get("cuobjdump_incomplete", 0))
        lines.extend(
            [
                f"  CUDA candidate outcomes ({platform}):",
                f"    Observed candidates: {value.get('cuda_candidate_files', 0)}",
                f"    Inspected with cuobjdump: {analyzed}",
                f"      - With embedded CUDA targets: {targets_found}",
                f"      - Without embedded device code: {no_device_code}",
                f"      - Incomplete inspections: {incomplete}",
                "    Precompiled kernel catalogs:",
                f"      - Catalog groups: {value.get('kernel_catalog_groups', 0)}",
                f"      - Shape-specific cubin members: {value.get('kernel_catalog_members', 0)}",
                (
                    "      - Members grouped as selectable alternatives: "
                    f"{value.get('kernel_catalog_grouped_members', 0)}"
                ),
                (
                    "      - Representative target variants: "
                    f"{value.get('kernel_catalog_target_variants', 0)}"
                ),
                "    Architecture-selected module catalogs:",
                f"      - Dispatch groups: {value.get('dispatch_module_groups', 0)}",
                f"      - Module variants: {value.get('dispatch_module_members', 0)}",
                (
                    "      - Target variants inspected: "
                    f"{value.get('dispatch_module_target_variants', 0)}"
                ),
                f"    Unverified target-string records: {value.get('marker_only_files', 0)}",
                "    Not inspected with cuobjdump:",
                f"      - Role-excluded: {value.get('skipped_non_runtime', 0)}",
                f"      - Candidate-content budget: {value.get('skipped_scan_budget', 0)}",
                f"      - Artifact-size limit: {value.get('skipped_artifact_limit', 0)}",
                f"      - File-count limit: {value.get('skipped_file_limit', 0)}",
                f"      - Analyzer unavailable: {value.get('skipped_no_cuobjdump', 0)}",
                "    Terms:",
                (
                    "      - Embedded CUDA targets: cuobjdump enumerated at least one cubin or "
                    "PTX target."
                ),
                (
                    "      - Without embedded device code: inspection completed but found no "
                    "cubin or PTX target."
                ),
                "      - Unverified target strings: raw string evidence only.",
                (
                    "      - Kernel catalog grouping: shape-specific cubins in the same catalog "
                    "and declared target bucket are selectable alternatives, so one "
                    "representative is inspected and duplicates are not coverage gaps."
                ),
                (
                    "      - Architecture-module grouping: recognized variants of the same "
                    "runtime-dispatched module family are evaluated together; unpaired modules "
                    "remain ordinary runtime artifacts."
                ),
            ]
        )
        reasons = value.get("cuobjdump_incomplete_reasons", {})
        if incomplete and isinstance(reasons, dict):
            lines.append(
                "    Incomplete inspection reasons: "
                f"{reasons.get('timeout', 0)} timeout, "
                f"{reasons.get('analyzer-error', 0)} analyzer error, "
                f"{reasons.get('io-error', 0)} I/O error, "
                f"{reasons.get('other', 0)} other."
            )
        max_candidate_bytes = settings.get("max_candidate_bytes")
        budget = (
            f"{int(max_candidate_bytes) / (1024**2):.0f} MiB"
            if isinstance(max_candidate_bytes, int)
            else "the configured --max-scan-mb value"
        )
        lines.extend(
            (
                "    Notes:",
                (
                    "      - Role-excluded: path heuristics classified the files as tests, "
                    "tooling, or host-side runtime."
                ),
                (
                    f"      - Candidate-content budget: cumulative {budget} read cap; this is "
                    "not the image size or host-memory limit."
                ),
                (
                    "      - Unverified target-string records: subset of observed candidates, "
                    "not additional files."
                ),
            )
        )
    return lines


def render_profiles_text(profiles: tuple[HardwareProfile, ...]) -> str:
    lines = ["Available hardware targets:"]
    for profile in profiles:
        aliases = ", ".join(profile.aliases)
        lines.append(
            f"  {profile.id:<8} sm_{profile.compute_capability_number:<3} "
            f"{profile.maturity:<11} {profile.display_name}"
        )
        lines.append(
            f"           CPU: {', '.join(profile.cpu_architectures)}; "
            f"CUDA >= {profile.minimum_cuda_version or 'unspecified'}; "
            f"NCCL >= {profile.minimum_nccl_version or 'unspecified'}; aliases: {aliases}"
        )
    return "\n".join(lines) + "\n"


def _finding_text(finding: Finding) -> list[str]:
    lines = [
        f"  [{SEVERITY_LABEL[finding.severity]}] {finding.id} ({finding.category}) {finding.title}",
        f"    Evidence: {finding.evidence}",
    ]
    if finding.details:
        lines.append("    Breakdown:")
        for detail in finding.details:
            context = f" ({detail.context})" if detail.context else ""
            lines.append(f"      - {detail.label}: {detail.value}{context}")
    lines.append(f"    Next step: {finding.remediation}")
    return lines


def render_text(report: InspectionReport) -> str:
    observations = report.observations
    lines = [
        "Nebius GPU Preflight",
        f"Source: {report.source}",
        (
            f"Report schema: v{report.schema_version}; tool: {report.tool.version}; "
            f"duration: {report.duration_seconds:.1f}s"
        ),
    ]
    static_limitations = observations.get("static_analysis_limitations", [])
    if isinstance(static_limitations, list) and static_limitations:
        lines.extend(["", "Static-analysis limitations:"])
        lines.extend(f"  - {item}" for item in static_limitations)
    for target in report.targets:
        actions = [
            *(
                finding
                for finding in report.global_findings
                if finding.severity in {"blocker", "warning"}
            ),
            *(finding for finding in target.findings if finding.severity in {"blocker", "warning"}),
        ]
        lines.extend(
            [
                "",
                f"{target.display_name} ({target.target})",
                f"  Compatibility: {target.compatibility_status}",
                "  CUDA artifact inspection coverage: "
                + _artifact_coverage_text(
                    target.artifact_inspection_coverage_percent,
                    target.artifact_inspection_covered_candidates,
                    target.artifact_inspection_eligible_candidates,
                ),
                f"  Reproducibility: {target.reproducibility_status}",
                (
                    f"  Profile: {target.profile_maturity}; "
                    f"schema v{target.profile_schema_version}; {target.profile_hash}"
                ),
                f"  Summary: {target.summary}",
                "  Actions:",
            ]
        )
        if actions:
            lines.extend(f"    - {finding.id}: {finding.remediation}" for finding in actions)
        else:
            lines.append("    - No static remediation required before hardware testing.")
        lines.append("  Passed checks:")
        if target.passed_checks:
            lines.extend(f"    - {item}" for item in target.passed_checks)
        else:
            lines.append("    - None established")
        lines.append("  Evidence limitations:")
        if target.evidence_limitations:
            lines.extend(f"    - {item}" for item in target.evidence_limitations)
        else:
            lines.append("    - See the top-level static-analysis limitations.")
        lines.extend(
            [
                "  Required hardware tests:",
                *(
                    f"    - {_hardware_check_text(item)}"
                    for item in target.required_hardware_checks
                ),
            ]
        )

    filesystem_scan = observations.get("image_scan_mode") == "filesystem"
    effective_targets = _effective_values(
        observations,
        "cuda_targets",
        "verified_image_cuda_targets" if filesystem_scan else "image_cuda_targets",
    )
    effective_cuda_versions = _cuda_release_versions(
        _effective_values(observations, "cuda_versions", "image_cuda_versions")
    )
    effective_nccl_versions = _effective_values(
        observations, "nccl_versions", "image_nccl_versions"
    )
    effective_packages = _effective_values(observations, "detected_packages", "image_packages")
    artifacts = observations.get("cuda_artifacts", [])
    marker_records = [item for item in artifacts if item.get("method") == "marker-scan"]
    runtime_marker_records = sum(item.get("role") == "runtime" for item in marker_records)
    unverified_targets = observations.get("unverified_image_cuda_target_strings", [])
    observed_lines = [
        "",
        "Observed evidence:",
        f"  Image references: {', '.join(observations['image_references']) or 'none detected'}",
        f"  Resolved platform: {observations['final_platform'] or 'unknown'}",
        f"  CUDA versions: {_compact(effective_cuda_versions)}",
        f"  NCCL versions: {_compact(effective_nccl_versions)}",
        f"  Packages: {_compact(effective_packages)}",
    ]
    observed_lines.insert(
        4,
        (
            f"  Verified embedded CUDA targets: {len(effective_targets)} unique"
            if filesystem_scan
            else f"  CUDA targets: {_compact(effective_targets)}"
        ),
    )
    if filesystem_scan:
        observed_lines.insert(
            5,
            f"  Unverified CUDA target strings: {len(unverified_targets)} unique across "
            f"{len(marker_records)} artifact(s); runtime-relevant artifacts: "
            f"{runtime_marker_records}",
        )
        target_summaries = observations.get("cuda_target_summaries", {})
        if isinstance(target_summaries, dict):
            offset = 5
            for target_name, summary in sorted(target_summaries.items()):
                if not isinstance(summary, dict):
                    continue
                base = str(summary.get("base_target", "target"))
                base_state = "present" if summary.get("base_target_present") else "absent"
                present = summary.get("specialized_targets_present", [])
                absent = summary.get("specialized_targets_absent", [])
                specialized = []
                if present:
                    specialized.append("present: " + ", ".join(present))
                if absent:
                    specialized.append("absent: " + ", ".join(absent))
                observed_lines.insert(
                    offset,
                    f"  Target relevance ({target_name}): native base {base} {base_state}; "
                    f"specialized targets {'; '.join(specialized) or 'none defined'}",
                )
                offset += 1
    lines.extend(observed_lines)
    lines.extend(_nccl_evidence_lines(observations, effective_nccl_versions))
    if observations.get("image_scan_mode"):
        analyzer = report.scan.capabilities.get("cuobjdump_version") or "unavailable"
        lines.extend(
            [
                f"  Image scan: {observations['image_scan_mode']}",
                "  Registry platforms: "
                + (", ".join(observations.get("available_platforms", [])) or "none detected"),
                f"  CUDA analyzer: {analyzer}",
            ]
        )
        installed = observations.get("installed_packages", [])
        if installed:
            lines.append(f"  Installed package records: {len(installed)} (first 200 retained)")
        if artifacts:
            targets_found = sum(
                item.get("method") == "cuobjdump"
                and item.get("complete")
                and bool(item.get("targets"))
                for item in artifacts
            )
            fallback = sum(item.get("method") == "marker-scan" for item in artifacts)
            incomplete = sum(
                item.get("method") == "cuobjdump" and not item.get("complete") for item in artifacts
            )
            lines.extend(
                (
                    "  Retained CUDA artifact evidence:",
                    f"    Total records: {len(artifacts)}",
                    f"    - With cuobjdump-enumerated targets: {targets_found}",
                    f"    - With unverified target strings: {fallback}",
                    f"    - Incomplete cuobjdump inspections: {incomplete}",
                )
            )
        lines.extend(_rootfs_statistics_lines(observations, report.scan.settings))
        skipped = observations.get("skipped_cuda_candidates", [])
        if skipped:
            lines.append("  Remaining uninspected runtime candidates:")
            for item in skipped[:50]:
                size_mib = int(item.get("size_bytes", 0)) / (1024**2)
                reason = SKIP_REASON_LABEL.get(str(item.get("reason")), str(item.get("reason")))
                lines.append(f"    - {item.get('path')} ({reason}; {size_mib:.1f} MiB)")
            if len(skipped) > 50:
                lines.append(f"    - {len(skipped) - 50} additional path(s) retained in JSON")
    if report.global_findings:
        lines.extend(["", "Global findings:"])
        for finding in report.global_findings:
            lines.extend(_finding_text(finding))
    for target in report.targets:
        lines.extend(
            [
                "",
                f"Detailed findings: {target.display_name} ({target.target})",
            ]
        )
        for finding in target.findings:
            lines.extend(_finding_text(finding))
    lines.extend(_cuda_compatibility_exception_lines(report))
    lines.extend(["", f"Note: {report.disclaimer}"])
    return "\n".join(lines) + "\n"


def render_json(report: InspectionReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"


def _finding_html(finding: Finding) -> str:
    detail_table = ""
    if finding.details:
        rows = "".join(
            "<tr>"
            f"<td>{escape(detail.label)}</td>"
            f"<td>{escape(str(detail.value))}</td>"
            f"<td>{escape(detail.context or '')}</td>"
            "</tr>"
            for detail in finding.details
        )
        detail_table = (
            '<div class="table-wrap finding-details"><table>'
            "<thead><tr><th>Breakdown</th><th>Value</th><th>Context</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    return (
        f'<article class="finding {escape(finding.severity)}">'
        f"<h3>{escape(SEVERITY_LABEL[finding.severity])} · {escape(finding.id)} · "
        f"{escape(finding.title)}</h3>"
        f"<p><strong>Category:</strong> {escape(finding.category)}</p>"
        f"<p><strong>Evidence:</strong> {escape(finding.evidence)}</p>"
        f"{detail_table}"
        f"<p><strong>Next step:</strong> {escape(finding.remediation)}</p>"
        "</article>"
    )


def _nccl_evidence_html(observations: dict[str, object], effective_versions: list[str]) -> str:
    evidence = observations.get("nccl_evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('version', 'unknown')))}</td>"
        f"<td>{escape(NCCL_SOURCE_LABEL.get(str(item.get('source')), str(item.get('source'))))}</td>"
        f"<td><code>{escape(str(item.get('name', 'unknown')))}</code></td>"
        f"<td><code>{escape(str(item.get('path') or ''))}</code></td>"
        "</tr>"
        for item in evidence
        if isinstance(item, dict)
    )
    if len(effective_versions) > 1:
        runtime_note = (
            "The effective loaded NCCL version is not statically established because multiple "
            "version sources were found. Confirm the loaded library during runtime validation."
        )
    else:
        runtime_note = (
            "Confirm the effective loaded NCCL version during runtime validation; static "
            "inventory does not prove dynamic-library selection."
        )
    return (
        "<section><h2>NCCL installations and version evidence</h2>"
        f"<p>{escape(runtime_note)}</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Version</th><th>Source</th><th>Record</th><th>Evidence path</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
    )


def _cuda_compatibility_exceptions_html(report: InspectionReport) -> str:
    exceptions = report.observations.get("cuda_compatibility_exceptions", [])
    groups = report.observations.get("cuda_compatibility_exception_groups", [])
    if not isinstance(exceptions, list) or not exceptions or not isinstance(groups, list):
        return ""
    display_names = {target.target: target.display_name for target in report.targets}
    kind_labels = {
        "kernel-catalog": "Precompiled cubin catalog",
        "architecture-module-group": "Architecture-variant module group",
        "runtime-artifact": "Runtime artifact",
    }
    summary_rows = "".join(
        "<tr>"
        f"<td>{escape(display_names.get(str(item.get('target')), str(item.get('target'))))}</td>"
        f"<td>{escape(str(item.get('component')))}</td>"
        f"<td>{escape(EXCEPTION_CLASSIFICATION_LABEL.get(str(item.get('classification')), str(item.get('classification'))))}</td>"
        f"<td>{escape(str(item.get('reason_summary')))}</td>"
        f"<td>{escape(str(item.get('exception_count')))}</td>"
        f"<td>{escape(str(item.get('file_count')))}</td>"
        "</tr>"
        for item in groups
        if isinstance(item, dict)
    )
    inventory_rows: list[str] = []
    for item in exceptions:
        if not isinstance(item, dict):
            continue
        paths = item.get("member_paths", [])
        if not isinstance(paths, list) or not paths:
            paths = [item.get("path")]
        for path in paths:
            inventory_rows.append(
                "<tr>"
                f"<td>{escape(display_names.get(str(item.get('target')), str(item.get('target'))))}</td>"
                f"<td>{escape(str(item.get('component')))}</td>"
                f"<td>{escape(EXCEPTION_CLASSIFICATION_LABEL.get(str(item.get('classification')), str(item.get('classification'))))}</td>"
                f"<td>{escape(kind_labels.get(str(item.get('kind')), 'Runtime artifact'))}</td>"
                f"<td><code>{escape(str(path))}</code></td>"
                f"<td>{escape(', '.join(str(value) for value in item.get('targets', [])) or 'none enumerated')}</td>"
                "</tr>"
            )
    rows = "".join(inventory_rows)
    return (
        "<section><h2>CUDA compatibility review summary</h2>"
        "<p>Counts describe binary files and grouped dispatch catalogs, not operations, "
        "features, or distinct workload code paths. Presence in the image does not prove that "
        "the workload loads an item.</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Target</th><th>Component</th><th>Assessment</th><th>Reason</th>"
        "<th>Review units</th><th>Binary files</th>"
        f"</tr></thead><tbody>{summary_rows}</tbody></table></div>"
        "<details><summary>Complete CUDA compatibility inventory "
        f"({len(inventory_rows)} binary file records)</summary>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Target</th><th>Component</th><th>Assessment</th><th>Type</th>"
        "<th>Image path</th><th>Embedded targets</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div></details></section>"
    )


def render_html(report: InspectionReport) -> str:
    observations = report.observations
    static_limitations = observations.get("static_analysis_limitations", [])
    static_limitations_html = ""
    if isinstance(static_limitations, list) and static_limitations:
        items = "".join(f"<li>{escape(str(item))}</li>" for item in static_limitations)
        static_limitations_html = (
            f"<section><h2>Static-analysis limitations</h2><ul>{items}</ul></section>"
        )
    filesystem_scan = observations.get("image_scan_mode") == "filesystem"
    effective_targets = _effective_values(
        observations,
        "cuda_targets",
        "verified_image_cuda_targets" if filesystem_scan else "image_cuda_targets",
    )
    effective_cuda_versions = _cuda_release_versions(
        _effective_values(observations, "cuda_versions", "image_cuda_versions")
    )
    effective_nccl_versions = _effective_values(
        observations, "nccl_versions", "image_nccl_versions"
    )
    effective_packages = _effective_values(observations, "detected_packages", "image_packages")
    nccl_evidence_section = _nccl_evidence_html(observations, effective_nccl_versions)
    cuda_exception_section = _cuda_compatibility_exceptions_html(report)
    artifacts = observations.get("cuda_artifacts", [])
    marker_records = [item for item in artifacts if item.get("method") == "marker-scan"]
    runtime_marker_records = sum(item.get("role") == "runtime" for item in marker_records)
    unverified_targets = observations.get("unverified_image_cuda_target_strings", [])
    image_summary = ""
    statistics_section = ""
    artifact_section = ""
    skipped_section = ""
    target_relevance_html = ""
    if observations.get("image_scan_mode"):
        analyzer = report.scan.capabilities.get("cuobjdump_version") or "unavailable"
        image_summary = (
            f"<div><strong>Image scan</strong><br>{escape(str(observations['image_scan_mode']))}</div>"
            f"<div><strong>Registry platforms</strong><br>"
            f"{escape(', '.join(observations.get('available_platforms', [])) or 'none detected')}</div>"
            f"<div><strong>CUDA analyzer</strong><br>{escape(str(analyzer))}</div>"
        )
        statistics = observations.get("rootfs_statistics", {})
        if isinstance(statistics, dict):
            platform_tables = []
            for platform, value in sorted(statistics.items()):
                if not isinstance(value, dict):
                    continue
                metrics = (
                    ("Observed CUDA candidates", value.get("cuda_candidate_files", 0), ""),
                    (
                        "Inspected with cuobjdump",
                        value.get("cuobjdump_files_analyzed", 0),
                        "",
                    ),
                    (
                        "With embedded CUDA targets",
                        value.get("cuobjdump_targets_found", 0),
                        "subset of inspected",
                    ),
                    (
                        "Without embedded device code",
                        value.get("cuobjdump_no_device_code", 0),
                        "subset of inspected",
                    ),
                    (
                        "Incomplete inspections",
                        value.get("cuobjdump_incomplete", 0),
                        "subset of inspected",
                    ),
                    (
                        "Kernel catalog groups",
                        value.get("kernel_catalog_groups", 0),
                        "precompiled dispatch catalogs",
                    ),
                    (
                        "Shape-specific catalog cubins",
                        value.get("kernel_catalog_members", 0),
                        "selectable alternatives",
                    ),
                    (
                        "Grouped catalog alternatives",
                        value.get("kernel_catalog_grouped_members", 0),
                        "not coverage gaps",
                    ),
                    (
                        "Representative catalog targets",
                        value.get("kernel_catalog_target_variants", 0),
                        "eligible inspection units",
                    ),
                    (
                        "Architecture-selected module groups",
                        value.get("dispatch_module_groups", 0),
                        "runtime dispatch families",
                    ),
                    (
                        "Architecture-selected module variants",
                        value.get("dispatch_module_members", 0),
                        "grouped alternatives",
                    ),
                    (
                        "Architecture-module target variants",
                        value.get("dispatch_module_target_variants", 0),
                        "eligible inspection units",
                    ),
                    (
                        "Unverified target-string records",
                        value.get("marker_only_files", 0),
                        "subset of observed candidates",
                    ),
                    (
                        "Role-excluded",
                        value.get("skipped_non_runtime", 0),
                        "tests, tooling, or host-side runtime",
                    ),
                    ("Scan budget", value.get("skipped_scan_budget", 0), "not inspected"),
                    (
                        "Artifact-size limit",
                        value.get("skipped_artifact_limit", 0),
                        "not inspected",
                    ),
                    (
                        "File-count limit",
                        value.get("skipped_file_limit", 0),
                        "not inspected",
                    ),
                    (
                        "Analyzer unavailable",
                        value.get("skipped_no_cuobjdump", 0),
                        "not inspected",
                    ),
                )
                rows = "".join(
                    "<tr>"
                    f"<td>{escape(str(label))}</td><td>{escape(str(count))}</td>"
                    f"<td>{escape(str(context))}</td>"
                    "</tr>"
                    for label, count, context in metrics
                )
                platform_tables.append(
                    f"<h3><code>{escape(str(platform))}</code></h3>"
                    '<div class="table-wrap"><table><thead><tr>'
                    "<th>Outcome</th><th>Count</th><th>Context</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table></div>"
                )
            if platform_tables:
                statistics_section = (
                    "<section><h2>CUDA candidate outcomes</h2>"
                    + "".join(platform_tables)
                    + "</section>"
                )
        if artifacts:
            rows = "".join(
                "<tr>"
                f"<td><code>{escape(str(item['path']))}</code></td>"
                f"<td>{escape(str(item['role']))}</td>"
                f"<td>{escape(str(item['method']))}</td>"
                f"<td>{'complete' if item['complete'] else 'incomplete: ' + str(item.get('incomplete_reason') or 'other')}</td>"
                f"<td>{escape(_compact(item['targets'], limit=12))}</td>"
                f"<td>{escape(str(item.get('diagnostic') or ''))}</td>"
                "</tr>"
                for item in artifacts
            )
            artifact_section = (
                "<section><h2>CUDA artifact evidence</h2>"
                "<p><strong>How to read this:</strong> An artifact with embedded CUDA targets had "
                "one or more cubin or PTX targets enumerated by cuobjdump. Marker-scan records "
                "contain unverified target-like strings and are discovery hints, not device-code "
                "proof. A CUDA artifact "
                "is a shared library, extension, or binary that may contain many GPU kernels and "
                "target images; the artifact itself is not one kernel.</p>"
                '<div class="table-wrap"><table><thead><tr>'
                "<th>Image path</th><th>Role</th><th>Method</th><th>Inspection status</th>"
                "<th>Targets</th><th>Diagnostic</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></div></section>"
            )
        skipped = observations.get("skipped_cuda_candidates", [])
        if skipped:
            rows = "".join(
                "<tr>"
                f"<td><code>{escape(str(item['path']))}</code></td>"
                f"<td>{escape(str(item['platform']))}</td>"
                f"<td>{escape(SKIP_REASON_LABEL.get(str(item['reason']), str(item['reason'])))}</td>"
                f"<td>{int(item['size_bytes']) / (1024**2):.1f} MiB</td>"
                "</tr>"
                for item in skipped
            )
            skipped_section = (
                "<section><h2>Remaining uninspected runtime candidates</h2>"
                "<p>Use these paths to decide whether a deeper rerun is relevant to the workload.</p>"
                '<div class="table-wrap"><table><thead><tr>'
                "<th>Image path</th><th>Platform</th><th>Reason</th><th>Size</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div></section>"
            )
        target_summaries = observations.get("cuda_target_summaries", {})
        if isinstance(target_summaries, dict):
            cards = []
            for target_name, summary in sorted(target_summaries.items()):
                if not isinstance(summary, dict):
                    continue
                base = str(summary.get("base_target", "target"))
                base_state = "present" if summary.get("base_target_present") else "absent"
                present = summary.get("specialized_targets_present", [])
                absent = summary.get("specialized_targets_absent", [])
                specialized = []
                if present:
                    specialized.append("present: " + ", ".join(present))
                if absent:
                    specialized.append("absent: " + ", ".join(absent))
                value = (
                    f"native base {base} {base_state}; specialized targets "
                    f"{'; '.join(specialized) or 'none defined'}"
                )
                cards.append(
                    f"<div><strong>Target relevance ({escape(str(target_name))})</strong><br>"
                    f"{escape(value)}</div>"
                )
            target_relevance_html = "".join(cards)
    target_sections = []
    for target in report.targets:
        actions = [
            *(
                finding
                for finding in report.global_findings
                if finding.severity in {"blocker", "warning"}
            ),
            *(finding for finding in target.findings if finding.severity in {"blocker", "warning"}),
        ]
        action_items = (
            "".join(
                f"<li><code>{escape(finding.id)}</code>: {escape(finding.remediation)}</li>"
                for finding in actions
            )
            or "<li>No static remediation required before hardware testing.</li>"
        )
        passed_items = (
            "".join(f"<li>{escape(item)}</li>" for item in target.passed_checks)
            or "<li>None established</li>"
        )
        limitation_items = (
            "".join(f"<li>{escape(item)}</li>" for item in target.evidence_limitations)
            or "<li>See the top-level static-analysis limitations.</li>"
        )
        hardware_items = "".join(
            f"<li><code>{escape(item)}</code>: "
            f"{escape(HARDWARE_CHECK_GUIDANCE.get(item, 'Execute and retain target-hardware evidence.'))}</li>"
            for item in target.required_hardware_checks
        )
        findings = "".join(_finding_html(finding) for finding in target.findings)
        target_sections.append(
            f'<section class="target"><h2>{escape(target.display_name)}</h2>'
            f'<div class="assessment"><div><strong>Compatibility</strong><br>'
            f'<span class="status">{escape(target.compatibility_status)}</span></div>'
            f"<div><strong>CUDA artifact inspection coverage</strong><br>"
            f"{escape(_artifact_coverage_text(target.artifact_inspection_coverage_percent, target.artifact_inspection_covered_candidates, target.artifact_inspection_eligible_candidates))}</div>"
            f"<div><strong>Reproducibility</strong><br>"
            f"{escape(target.reproducibility_status)}</div></div>"
            f'<p class="lead">{escape(target.summary)}</p>'
            f"<h3>Actions</h3><ul>{action_items}</ul>"
            f"<h3>Passed checks</h3><ul>{passed_items}</ul>"
            f"<h3>Evidence limitations</h3><ul>{limitation_items}</ul>"
            f"<h3>Required hardware tests</h3><ul>{hardware_items}</ul>"
            f"<details><summary>Technical findings</summary>{findings}</details>"
            f"<p>Profile: <code>{escape(target.target)}</code> · "
            f"{escape(target.profile_maturity)} · schema {target.profile_schema_version} · "
            f"<code>{escape(target.profile_hash)}</code></p></section>"
        )
    global_section = ""
    if report.global_findings:
        global_section = (
            "<section><h2>Global findings</h2>"
            + "".join(_finding_html(finding) for finding in report.global_findings)
            + "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nebius GPU Preflight report</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, system-ui, sans-serif; }}
    body {{ max-width: 1100px; margin: 0 auto; padding: 2rem; line-height: 1.5; }}
    header, section {{ margin-bottom: 2rem; }}
    .summary, .assessment {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: .8rem; }}
    .summary div, .assessment div, .finding {{ border: 1px solid #7776; border-radius: .6rem; padding: .8rem 1rem; }}
    .finding {{ margin: .8rem 0; border-left-width: .4rem; }}
    .finding.blocker {{ border-left-color: #dc2626; }}
    .finding.warning {{ border-left-color: #d97706; }}
    .finding.info {{ border-left-color: #2563eb; }}
    .finding h3 {{ font-size: 1rem; margin: 0 0 .5rem; }}
    .finding p {{ margin: .35rem 0; }}
    .status {{ font-size: .75rem; border: 1px solid currentColor; border-radius: 1rem; padding: .15rem .55rem; }}
    .lead {{ font-size: 1.1rem; }}
    details summary {{ cursor: pointer; font-weight: 700; margin: 1rem 0; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
    th, td {{ border: 1px solid #7776; padding: .45rem; text-align: left; vertical-align: top; }}
    code {{ overflow-wrap: anywhere; }}
    footer {{ border-top: 1px solid #7776; padding-top: 1rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Nebius GPU Preflight</h1>
    <p>Static readiness report for <code>{escape(report.source)}</code></p>
    <p>Schema v{report.schema_version} · tool {escape(report.tool.version)} ·
    {report.duration_seconds:.1f}s</p>
  </header>
  {static_limitations_html}
  {"".join(target_sections)}
  <section>
    <h2>Observed stack</h2>
    <div class="summary">
      <div><strong>Image references</strong><br>{escape(", ".join(observations["image_references"]) or "none detected")}</div>
      <div><strong>Resolved platform</strong><br>{escape(str(observations["final_platform"] or "unknown"))}</div>
      <div><strong>{"Verified embedded CUDA targets" if filesystem_scan else "CUDA targets"}</strong><br>{escape(f"{len(effective_targets)} unique" if filesystem_scan else _compact(effective_targets))}</div>
      {f"<div><strong>Unverified CUDA target strings</strong><br>{len(unverified_targets)} unique across {len(marker_records)} artifact(s); runtime-relevant artifacts: {runtime_marker_records}</div>" if filesystem_scan else ""}
      <div><strong>CUDA versions</strong><br>{escape(_compact(effective_cuda_versions))}</div>
      <div><strong>NCCL versions</strong><br>{escape(_compact(effective_nccl_versions))}</div>
      <div><strong>Packages</strong><br>{escape(_compact(effective_packages))}</div>
      {target_relevance_html}
      {image_summary}
    </div>
  </section>
  {nccl_evidence_section}
  {statistics_section}
  {cuda_exception_section}
  {artifact_section}
  {skipped_section}
  {global_section}
  <footer><strong>Scope:</strong> {escape(report.disclaimer)}</footer>
</body>
</html>
"""
