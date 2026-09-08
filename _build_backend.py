"""Build-only offline knowledge producer; never imports SCGSim or a solver.

Canonical usage remains in the listed QMD sources. The generated package data
implements scq-mcp-akb-1.0, not a runtime API or a second editable recipe set.
Setuptools retains ownership of all distribution metadata and wheel RECORDs.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from setuptools import build_meta as _setuptools
from setuptools.build_meta import *  # noqa: F403 (forward unchanged PEP 517/660 hooks)

# IDs survive source moves. Descriptions index sources; no procedure is copied here.
_RESOURCES = (
    (
        "guide/overview",
        "docs/agent-knowledge.qmd",
        "guide",
        "Package knowledge, prerequisites, and canonical resource inventory.",
    ),
    (
        "capabilities/backends",
        "docs/backend-support.qmd",
        "capability",
        "Implemented candidates, unimplemented solvers, and explicit exclusions.",
    ),
    (
        "guide/sgb",
        "docs/geometry-sgb.qmd",
        "guide",
        "Structured component/PDK inputs and Route A/B semantic geometry.",
    ),
    (
        "workflow/notebook",
        "docs/notebook-ux.qmd",
        "workflow",
        "Stage-local API calls, immutable runs, and consumer Notebook boundaries.",
    ),
    (
        "api/palace",
        "docs/specs/palace-electrostatic-sgb.qmd",
        "api",
        "Palace Electrostatic/Eigenmode preparation, grounds, routes, and Surface EPR.",
    ),
    (
        "api/aedt",
        "docs/specs/aedt-runtime.qmd",
        "api",
        "HFSS/Q3D/Q2D typed setup, native exports, convergence, and limitations.",
    ),
    (
        "guide/execution",
        "docs/execution-profiles.qmd",
        "guide",
        "Manual handoff, launch profiles, and preparation versus execution.",
    ),
    (
        "guide/reports",
        "docs/report-model.qmd",
        "guide",
        "Trust-first reports, strict/partial resolution, masks, loss/Q, and nonclaims.",
    ),
    (
        "guide/preview",
        "docs/geometry-preview.qmd",
        "guide",
        "Semantic geometry inspection, rendering prerequisites, and unavailable modes.",
    ),
    (
        "guide/provenance",
        "docs/provenance.qmd",
        "guide",
        "Exact receipt/hash identity and public data boundaries.",
    ),
    (
        "guide/upstream",
        "docs/goals-and-upstream.qmd",
        "guide",
        "Independent downstream scope, non-goals, and derivation provenance.",
    ),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _generate_knowledge() -> None:
    root = Path(__file__).resolve().parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    bundle = root / "src/scgsim/_agent_knowledge"
    resources = []
    outputs: dict[str, bytes] = {}
    for resource_id, source_ref, kind, summary in sorted(_RESOURCES):
        source = root / source_ref
        if source.resolve() != source or not source.is_file():
            raise ValueError(
                f"Knowledge source must be a regular non-symlink file: {source_ref}"
            )
        content = source.read_bytes()
        text = content.decode("utf-8")
        # The selected canonical QMD pages have single-line JSON-quoted YAML titles.
        title_line = re.search(r'^title: ("[^\n]+")$', text, re.MULTILINE)
        if title_line is None:
            raise ValueError(f"Knowledge source needs a quoted title: {source_ref}")
        title = json.loads(title_line.group(1))
        digest = hashlib.sha256(content).hexdigest()
        outputs[source_ref] = content
        resources.append(
            {
                "id": resource_id,
                "kind": kind,
                "title": title,
                "summary": summary,
                "status": (
                    "CONVERGING usage knowledge; backend-specific availability "
                    "and nonclaims are explicit in the content"
                ),
                "path": source_ref,
                "content_sha256": digest,
                "provenance": {"source_ref": source_ref, "source_sha256": digest},
            }
        )
    manifest = {
        "schema_version": "1.0",
        "distribution": project["name"],
        "distribution_version": project["version"],
        "resources": resources,
    }
    manifest["bundle_id"] = "sha256:" + hashlib.sha256(_canonical(manifest)).hexdigest()
    outputs["manifest.json"] = _canonical(manifest) + b"\n"

    # Never follow a generated-directory symlink or silently package stray files.
    for relative in outputs:
        target = bundle / relative
        if target.resolve() != target:
            raise ValueError("Knowledge output must not traverse a symlink")
    if bundle.exists():
        unexpected = {
            item.relative_to(bundle).as_posix()
            for item in bundle.rglob("*")
            if item.is_symlink()
            or (
                not item.is_dir() and item.relative_to(bundle).as_posix() not in outputs
            )
        }
        if unexpected:
            raise ValueError(
                "Unexpected files in generated knowledge directory; "
                "preserve and inspect them before rebuilding"
            )
    for relative, content in outputs.items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != content:
            target.write_bytes(content)


def build_sdist(sdist_directory, config_settings=None):
    _generate_knowledge()
    return _setuptools.build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _generate_knowledge()
    return _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)
