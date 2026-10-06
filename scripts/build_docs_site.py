"""Render immutable documentation sources into one versioned Pages publication.

Published content comes from frozen Git archives. An explicit clean export is
supported for unpublished development previews, with its own inventory identity.
Historical bodies remain original; only publisher-owned presentation is overlaid.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
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
PREFIX = "/scgsim/"
QUARTO_VERSION = "1.10.18"
ASKR_COMMIT = "611f537ca3ec7089f7b5daf983f1554f8c98a8dd"


def resolve_branches() -> dict[str, str]:
    """Capture both branch heads in one remote lookup."""
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


def export_inventory(source: Path) -> dict[str, str]:
    """Bind an explicit export without attributing it to a published commit."""
    if (source / ".git").exists() or (source / "_site").exists() or (source / ".quarto").exists():
        raise ValueError("candidate must be a clean source export, not a checkout or render")
    rows = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"candidate export contains a symbolic link: {path}")
        if path.is_file():
            rows[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def package_version(source: Path) -> str:
    return tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]


def version_line(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:\.dev\d+|rc\d+)?", version)
    if match is None:
        raise ValueError(f"unsupported documentation package version: {version!r}")
    return match[1]


def presentation_blocks(config: str) -> dict[str, str]:
    """Select the two existing top-level Quarto presentation mappings."""
    starts = list(re.finditer(r"(?m)^([A-Za-z][\w-]*):", config))
    blocks = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(config)
        blocks[match[1]] = config[match.start():end]
    return {name: blocks[name] for name in ("website", "format")}


def historical_website(website: str, source: Path) -> str:
    """Apply the publisher menu only to pages owned by this historical source."""
    # Area landings follow the first available entry in that same current sidebar.
    # The publisher configuration remains the sole maintained navigation authority.
    sidebars = list(re.finditer(r"(?m)^    - id: ([^\n]+)\n", website))
    landings = {}
    for index, match in enumerate(sidebars):
        end = sidebars[index + 1].start() if index + 1 < len(sidebars) else len(website)
        pages = re.findall(r"(?m)^\s+- (\S+\.qmd)\s*$", website[match.end():end])
        landings[match[1]] = next((page for page in pages if (source / page).is_file()), None)

    def landing(match: re.Match[str]) -> str:
        area, page = match[1], match[2]
        if (source / page).is_file():
            return match[0]
        replacement = landings[area.lower()]
        if replacement is None:
            raise RuntimeError(f"historical source has no page for the {area} Area")
        return match[0].replace(page, replacement)

    website = re.sub(
        r'(?m)^      - text: "([^"\n]+)"\n        href: (\S+\.qmd)\s*$',
        landing, website,
    )
    return "".join(
        line for line in website.splitlines(keepends=True)
        if not (match := re.fullmatch(r"\s+- (\S+\.qmd)\s*", line))
        or (source / match[1]).is_file()
    )


def overlay_presentation(source: Path, presentation: Path) -> None:
    current = presentation_blocks((presentation / "_quarto.yml").read_text())
    current["website"] = historical_website(current["website"], source)
    original = (source / "_quarto.yml").read_text()
    blocks = presentation_blocks(original)
    for name in ("website", "format"):
        original = original.replace(blocks[name], current[name], 1)
    (source / "_quarto.yml").write_text(original)
    # Native Askr presentation supersedes the historical consumer stylesheet.
    (source / "docs/styles.css").unlink(missing_ok=True)
    vendor = source / "_extensions/arfiligol/askr"
    if vendor.exists():
        shutil.rmtree(vendor)
    shutil.copytree(presentation / "_extensions/arfiligol/askr", vendor)


def redirect_page(destination: str) -> str:
    safe = html.escape(destination, quote=True)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>SCGSim documentation</title><meta http-equiv="refresh" content="0;url={safe}">'
        f'</head><body><p>Open <a href="{safe}">SCGSim documentation</a>.</p>'
        '<script>location.replace(' + json.dumps(destination)
        + '+location.search+location.hash);</script></body></html>'
    )


def missing_page(roots: list[str], aliases: dict[str, str], latest: str) -> str:
    routes = json.dumps({"roots": roots, "aliases": aliases, "latest": latest}).replace("<", "\\u003c")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>SCGSim documentation page not found</title></head><body><main>'
        '<h1>Documentation page not found</h1>'
        '<p id="reason">This path is not present in the published documentation.</p>'
        f'<p><a href="{html.escape(latest, quote=True)}">Open latest stable documentation</a></p>'
        '</main><script>const routes=' + routes + ";" + r"""
const path = location.pathname;
function within(root) { return path === root.slice(0, -1) || path.startsWith(root); }
const alias = Object.keys(routes.aliases).find(within);
if (alias) {
  const rest = path === alias.slice(0, -1) ? '' : path.slice(alias.length);
  location.replace(routes.aliases[alias] + rest + location.search + location.hash);
} else {
  const root = routes.roots.find(within);
  if (root) location.replace(root + location.search + location.hash);
}
</script></body></html>"""
    )


def display_identity(value: str) -> str:
    """Keep complete visible identities while allowing native HTML line wrapping."""
    return "<wbr>".join(html.escape(value[index:index + 16]) for index in range(0, len(value), 16))


def render_source(source: Path, entry: dict, presentation_identity: dict, quarto: str) -> None:
    root = entry["root"]
    content = entry.get("content_commit")
    if content:
        attribution = f'content <a href="{REPO_URL}/tree/{content}">{display_identity(content)}</a>'
    else:
        attribution = "unpublished candidate · export " + display_identity(entry["candidate_export_sha256"])
    presentation = presentation_identity.get("presentation_commit")
    if presentation:
        display = f'presentation <a href="{REPO_URL}/tree/{presentation}">{display_identity(presentation)}</a>'
    else:
        display = "unpublished presentation · export " + display_identity(presentation_identity["presentation_export_sha256"])
    website = {
        "site-url": "https://orpenstrike.github.io" + root,
        "page-footer": {"center": f'SCGSim {entry["package_version"]} · {attribution} · {display} · Askr {display_identity(ASKR_COMMIT)}'},
    }
    if content:
        website["repo-branch"] = content
    else:
        # Preview links must not pretend the unpublished bodies exist on GitHub.
        website["repo-actions"] = ["issue"]
    # Quarto's repository-link postprocessor reads the base project mapping.
    # Bind these actions there as well as in the render profile.
    config_path = source / "_quarto.yml"
    config = config_path.read_text()
    config = re.sub(r"(?m)^  repo-actions:.*$",
                    "  repo-actions: [source, issue]" if content else "  repo-actions: [issue]", config)
    if content:
        config = re.sub(r"(?m)^  repo-branch:.*\n", "", config)
        config = re.sub(r"(?m)^(  repo-url:.*)$", r"\1\n  repo-branch: " + content, config)
    config_path.write_text(config)
    profile = source / "_quarto-pages.yml"
    if profile.exists():
        raise RuntimeError("source already owns the Pages profile name")
    profile.write_text(json.dumps({"project": {"output-dir": "_site"}, "execute": {"enabled": False}, "website": website}))
    subprocess.run([quarto, "render", str(source), "--profile", "pages", "--no-execute"], check=True)
    if not (source / "_site/index.html").is_file():
        raise RuntimeError("render did not produce its home page")


def build(branches: dict[str, str], output: Path, quarto: str = "quarto", *,
          catalog: Path | None = None, candidate_source: Path | None = None,
          presentation_commit: str | None = None) -> None:
    actual = subprocess.run([quarto, "--version"], check=True, capture_output=True, text=True)
    if actual.stdout.strip() != QUARTO_VERSION:
        raise RuntimeError(f"Pages requires Quarto {QUARTO_VERSION}")
    catalog = catalog or Path(__file__).with_name("pages_versions.json")
    policy = json.loads(catalog.read_text())
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Presentation has an explicit immutable authority; preview uses its exact export.
    if candidate_source is None and presentation_commit is None:
        presentation_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
        ).strip()
    with tempfile.TemporaryDirectory(prefix="scgsim-pages-", dir=output.parent) as temporary:
        work = Path(temporary)
        if candidate_source is not None:
            rows = export_inventory(candidate_source)
            digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            development = work / "candidate"
            shutil.copytree(candidate_source, development)
            presentation = development
            presentation_identity = {"presentation_export_sha256": digest}
            development_identity = {"candidate_export_sha256": digest, "candidate_inventory": rows}
        else:
            development = snapshot(branches[policy["development_ref"]], work / "development")
            presentation = snapshot(presentation_commit, work / "presentation")
            presentation_identity = {"presentation_commit": presentation_commit}
            development_identity = {"content_commit": branches[policy["development_ref"]]}
        entries = []
        sources = {}

        def add(source: Path, version: str, identity: dict, branch: str) -> None:
            prerelease = branch == policy["development_ref"]
            slug = "v" + version_line(version) + "-dev" if prerelease else "v" + version
            existing = next((item for item in entries if item["slug"] == slug), None)
            if existing:
                if existing.get("content_commit") != identity.get("content_commit"):
                    raise ValueError("one version root has conflicting content commits")
                existing["branches"].append(branch)
                return
            entry = {"slug": slug, "root": PREFIX + slug + "/", "package_version": version,
                     "label": version_line(version) + " dev" if prerelease else version,
                     "branches": [branch], **identity}
            entries.append(entry)
            sources[slug] = source

        add(development, package_version(development), development_identity, policy["development_ref"])
        stable_ref = policy["stable_ref"]
        if stable_ref in branches:
            stable = snapshot(branches[stable_ref], work / "stable")
            version = package_version(stable)
            if version != version_line(version):
                raise ValueError("current stable source has prerelease metadata")
            add(stable, version, {"content_commit": branches[stable_ref]}, stable_ref)
        for pinned in policy["historical"]:
            existing = next((item for item in entries
                             if item.get("content_commit") == pinned["commit"]
                             and item["package_version"] == pinned["version"]), None)
            if existing:
                existing["branches"].append("historical")
                continue
            historical = snapshot(pinned["commit"], work / ("history-" + pinned["version"]))
            if package_version(historical) != pinned["version"]:
                raise ValueError("historical content version differs from the catalogue")
            add(historical, pinned["version"], {"content_commit": pinned["commit"]}, "historical")
        stable_entries = [item for item in entries if policy["development_ref"] not in item["branches"]]
        latest = max(stable_entries, key=lambda item: tuple(map(int, item["package_version"].split("."))))
        aliases = {PREFIX + "develop/": entries[0]["root"], PREFIX + "main/": latest["root"]}
        manifest = {"mode": "inline", "versions": [
            {"label": item["label"], "root": item["root"], **({"latest": True} if item is latest else {})}
            for item in entries
        ]}
        site = work / "site"
        site.mkdir()
        for entry in entries:
            source = sources[entry["slug"]]
            if "historical" in entry["branches"]:
                overlay_presentation(source, presentation)
                actual_presentation = presentation_identity
            elif "content_commit" in entry:
                actual_presentation = {"presentation_commit": entry["content_commit"]}
            else:
                actual_presentation = {"presentation_export_sha256": entry["candidate_export_sha256"]}
            entry.update(actual_presentation)
            render_source(source, entry, actual_presentation, quarto)
            destination = site / entry["slug"]
            shutil.copytree(source / "_site", destination)
            (destination / "askr-versions.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (site / "index.html").write_text(redirect_page(latest["root"]))
        for alias, destination in aliases.items():
            target = site / alias.removeprefix(PREFIX)
            target.mkdir()
            (target / "index.html").write_text(redirect_page(destination))
        (site / "404.html").write_text(missing_page([item["root"] for item in entries], aliases, latest["root"]))
        (site / "build.json").write_text(json.dumps({
            "repository": REPOSITORY, "quarto_version": QUARTO_VERSION,
            "askr_commit": ASKR_COMMIT, **presentation_identity,
            "branches": branches, "versions": entries, "latest_stable": latest["root"],
            "preview": candidate_source is not None,
        }, indent=2) + "\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        # No combined output is exposed until every required render completes.
        os.rename(site, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--quarto", default="quarto")
    parser.add_argument("--candidate-source", type=Path,
                        help="explicit clean exported development preview, never published-commit attribution")
    parser.add_argument("--presentation-commit")
    arguments = parser.parse_args()
    build(resolve_branches(), arguments.output, arguments.quarto,
          catalog=arguments.catalog, candidate_source=arguments.candidate_source,
          presentation_commit=arguments.presentation_commit)
