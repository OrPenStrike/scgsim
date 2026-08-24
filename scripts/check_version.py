"""Validate SCGSim package metadata against the selected repository line."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?P<dev>\.dev\d+)|(?P<rc>rc\d+))?$"
)


def version_kind(version: str) -> str:
    """Return the SCGSim release class encoded by a PEP 440 version."""

    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported SCGSim version: {version!r}")
    if match.group("dev"):
        return "development"
    if match.group("rc"):
        return "release-candidate"
    return "stable"


def validate_line(version: str, line: str) -> None:
    """Require prereleases on develop and stable releases on main."""

    kind = version_kind(version)
    if line == "develop" and kind == "stable":
        raise ValueError("develop must use a development or release-candidate version")
    if line == "main" and kind != "stable":
        raise ValueError("main must use a stable version")


def repository_version(root: Path) -> str:
    """Return one version shared by pyproject.toml and uv.lock."""

    with (root / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    with (root / "uv.lock").open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    locked = [
        package["version"]
        for package in packages
        if package.get("name") == "scgsim"
        and package.get("source") == {"editable": "."}
    ]
    if locked != [version]:
        raise ValueError(f"scgsim lock version {locked!r} does not match {version!r}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=("any", "develop", "main"), default="any")
    args = parser.parse_args()
    version = repository_version(Path(__file__).resolve().parents[1])
    if args.line != "any":
        validate_line(version, args.line)
    else:
        version_kind(version)
    print(f"scgsim {version} ({args.line})")


if __name__ == "__main__":
    main()
