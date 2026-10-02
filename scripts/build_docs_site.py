"""Build the public develop/main documentation snapshots for one Pages artifact.

Only remote Git commit archives supply source files. Rendering never executes
document code; absent main is distinct from a failed remote lookup or build.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path

REPOSITORY = "OrPenStrike/scgsim"
REPO_URL = f"https://github.com/{REPOSITORY}"
SITE_URL = "https://orpenstrike.github.io/scgsim/"
QUARTO_VERSION = "1.10.18"


def resolve_branches() -> dict[str, str]:
    # One lookup freezes both heads when this queued build starts.
    result = subprocess.run(
        ["git", "ls-remote", "--heads", f"{REPO_URL}.git",
         "refs/heads/develop", "refs/heads/main"],
        check=True, capture_output=True, text=True,
    )
    branches = {}
    for line in result.stdout.splitlines():
        sha, ref = line.split()
        if ref not in {"refs/heads/develop", "refs/heads/main"} or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("remote returned an invalid commit identity")
        branches[ref.removeprefix("refs/heads/")] = sha
    if "develop" not in branches:
        raise RuntimeError("remote develop branch is missing")
    return branches


def snapshot(sha: str, directory: Path) -> Path:
    directory.mkdir()
    archive = directory / "source.tar.gz"
    url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{sha}"
    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")
    archive.unlink()
    root, = directory.iterdir()
    return root


def render_branch(
    root: Path, branch: str, sha: str, branches: dict[str, str], quarto: str,
) -> dict[str, str]:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    navigation = [{"text": "Versions", "href": SITE_URL}]
    navigation.extend({"text": name, "href": f"{SITE_URL}{name}/"} for name in branches)
    footer = (
        f"{branch} · SCGSim {html.escape(version)} · "
        f'<a href="{REPO_URL}/commit/{sha}">source {sha[:12]}</a>'
    )
    profile = root / "_quarto-pages.yml"
    if profile.exists():
        raise RuntimeError("snapshot already owns the Pages profile name")
    profile.write_text(json.dumps({
        "project": {"output-dir": "_site"},
        "execute": {"enabled": False},
        "website": {
            "site-url": f"{SITE_URL}{branch}/", "repo-branch": branch,
            "navbar": {"right": navigation}, "page-footer": {"center": footer},
        },
    }))
    subprocess.run(
        [quarto, "render", str(root), "--profile", "pages", "--no-execute"],
        check=True,
    )
    if not (root / "_site/index.html").is_file():
        raise RuntimeError(f"{branch} render did not produce its home page")
    return {"commit": sha, "package_version": version, "site_url": f"{SITE_URL}{branch}/"}


def build(branches: dict[str, str], output: Path, quarto: str = "quarto") -> None:
    actual = subprocess.run([quarto, "--version"], check=True, capture_output=True, text=True)
    if actual.stdout.strip() != QUARTO_VERSION:
        raise RuntimeError(f"Pages requires Quarto {QUARTO_VERSION}")
    # Publishable output appears only after every available snapshot succeeds.
    with tempfile.TemporaryDirectory(prefix="scgsim-pages-") as temporary:
        work = Path(temporary)
        site = work / "site"
        site.mkdir()
        sources = {}
        for branch, sha in branches.items():
            root = snapshot(sha, work / branch)
            sources[branch] = render_branch(root, branch, sha, branches, quarto)
            shutil.copytree(root / "_site", site / branch)
        items = []
        for branch, source in sources.items():
            items.append(
                f'<li><a href="{branch}/">{branch}</a> · '
                f'SCGSim {html.escape(source["package_version"])} · '
                f'<a href="{REPO_URL}/commit/{source["commit"]}">'
                f'source {source["commit"][:12]}</a></li>'
            )
        if "main" not in sources:
            items.append("<li>main — not yet published</li>")
        (site / "index.html").write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>SCGSim documentation versions</title><style>'
            ':root{color-scheme:light dark}body{font:1.1rem/1.6 system-ui;'
            'max-width:48rem;margin:4rem auto;padding:0 1rem}'
            'a{color:LinkText}li{margin:1rem 0}</style></head><body><main>'
            '<h1>SCGSim documentation</h1><p>Choose a documentation branch. '
            'Each site records its package version and source commit.</p><ul>'
            + "".join(items) + '</ul><p><a href="build.json">Build sources</a></p>'
            '</main></body></html>'
        )
        (site / "build.json").write_text(json.dumps({
            "repository": REPOSITORY, "quarto_version": QUARTO_VERSION, "branches": sources,
        }, indent=2) + "\n")
        shutil.copytree(site, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(resolve_branches(), arguments.output)
