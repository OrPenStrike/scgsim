"""Generate the installed offline knowledge bundle before distribution builds.

Canonical page content and producer metadata stay in README.md and the
authored Quarto pages. This build-only adapter writes ordinary package data;
it does not import SCGSim or add a runtime producer API.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from setuptools import build_meta as _setuptools
from setuptools.build_meta import *  # noqa: F403 (forward remaining PEP 517 hooks)

_PAGE_FIELDS = {
    "scgsim-id": "id",
    "scgsim-kind": "kind",
    "title": "title",
    "description": "summary",
    "scgsim-status": "status",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _frontmatter(
    content: bytes, source_ref: str, required: tuple[str, ...]
) -> dict[str, str]:
    text = content.decode("utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"Knowledge source needs YAML front matter: {source_ref}")

    values: dict[str, str] = {}
    closed = False
    for line in lines[1:]:
        if line == "---":
            closed = True
            break
        key, separator, raw_value = line.partition(":")
        if not separator or key not in required:
            continue
        if key in values:
            raise ValueError(f"Knowledge source repeats front matter field {key!r}: {source_ref}")
        value = json.loads(raw_value.strip())
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Knowledge source field {key!r} must be a nonempty string: {source_ref}"
            )
        values[key] = value

    if not closed:
        raise ValueError(f"Knowledge source has unterminated YAML front matter: {source_ref}")
    missing = set(required).difference(values)
    if missing:
        raise ValueError(
            f"Knowledge source is missing front matter fields {sorted(missing)!r}: {source_ref}"
        )
    return values


def _source_bytes(source: Path, source_ref: str) -> bytes:
    if source.resolve() != source or not source.is_file():
        raise ValueError(f"Knowledge source must be a regular non-symlink file: {source_ref}")
    return source.read_bytes()


def _resource(
    source_ref: str,
    output_path: str,
    content: bytes,
    *,
    identifier: str,
    kind: str,
    title: str,
    summary: str,
    status: str,
) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    return {
        "id": identifier,
        "kind": kind,
        "title": title,
        "summary": summary,
        "status": status,
        "path": output_path,
        "content_sha256": digest,
        "provenance": {"source_ref": source_ref, "source_sha256": digest},
    }


def _generate_knowledge() -> None:
    root = Path(__file__).resolve().parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    outputs: dict[str, bytes] = {}
    resources: list[dict[str, object]] = []

    readme_ref = "README.md"
    readme_path = root / readme_ref
    readme = _source_bytes(readme_path, readme_ref)
    readme_title = ""
    for line in readme.decode("utf-8").splitlines():
        if line.startswith("# "):
            readme_title = line[2:].strip()
            break
    if not readme_title:
        raise ValueError(f"Knowledge README is missing a Markdown H1 title: {readme_ref}")
    outputs[readme_ref] = readme
    resources.append(
        _resource(
            readme_ref,
            readme_ref,
            readme,
            identifier="readme",
            kind="project",
            title=readme_title,
            summary=project["description"],
            status="STABILIZED",
        )
    )

    docs = root / "docs"
    if docs.resolve() != docs or not docs.is_dir():
        raise ValueError("Knowledge source directory must be a regular non-symlink directory: docs")
    pages = sorted(docs.rglob("*.qmd"), key=lambda path: path.relative_to(root).as_posix())
    for source in pages:
        source_ref = source.relative_to(root).as_posix()
        content = _source_bytes(source, source_ref)
        frontmatter = _frontmatter(content, source_ref, tuple(_PAGE_FIELDS))
        metadata = {
            manifest_field: frontmatter[field]
            for field, manifest_field in _PAGE_FIELDS.items()
        }
        output_path = source_ref
        outputs[output_path] = content
        resources.append(
            _resource(
                source_ref,
                output_path,
                content,
                identifier=metadata["id"],
                kind=metadata["kind"],
                title=metadata["title"],
                summary=metadata["summary"],
                status=metadata["status"],
            )
        )

    identifiers = [resource["id"] for resource in resources]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Knowledge resource IDs must be unique")
    resources.sort(key=lambda resource: resource["id"])
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "distribution": project["name"],
        "distribution_version": project["version"],
        "resources": resources,
    }
    manifest["bundle_id"] = "sha256:" + hashlib.sha256(_canonical(manifest)).hexdigest()
    outputs["manifest.json"] = _canonical(manifest) + b"\n"

    bundle = root / "src/scgsim/_agent_knowledge"
    for relative in outputs:
        target = bundle / relative
        if target.resolve() != target:
            raise ValueError("Knowledge output must not traverse a symlink")
    if bundle.exists():
        unexpected = {
            item.relative_to(bundle).as_posix()
            for item in bundle.rglob("*")
            if item.is_symlink()
            or (not item.is_dir() and item.relative_to(bundle).as_posix() not in outputs)
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
