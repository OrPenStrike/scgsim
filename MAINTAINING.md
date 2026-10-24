# Maintaining SCGSim

This repository reference covers documentation builds and package maintenance.
The secondary [maintainer guide](docs/ownership.qmd) links the implementation
and project references. Researchers can start with the
[course](docs/examples.qmd) without a parent workspace.

## Sources and generated output

`src/scgsim/` owns reusable product code. `docs/` and README own authored
explanations; `_quarto.yml` owns site navigation and execution settings.
`scripts/` owns build/version tools; `tests/` owns maintained package tests.
Generated HTML, search indexes and packaged knowledge are derived output.
Public component and executable notebook sources belong to OrPen SC PDK.

## Native presentation

The site uses the complete upstream Askr extension without local CSS or JS.
The current authoring pin is Askr v0.4.2 at
`611f537ca3ec7089f7b5daf983f1554f8c98a8dd`, requiring Quarto 1.10.18.
A later formal producer update must retain its own exact version/commit.
Use native Quarto structures for tables, callouts and navigation; keep source
identity separate from generated presentation identity.

## Documentation builds

`scripts/pages_versions.json` is the version catalogue.
`scripts/build_docs_site.py` freezes source revisions and metadata, renders
clean snapshots with code execution disabled and assembles the combined site.
Current main/develop use their own content; retained historical versions keep
original README/Quarto bodies with a documented presentation overlay.

The root selects latest stable. Version changes retain the current page where
it exists, otherwise use that version's home. `/main/` and `/develop/` are aliases.
The inline catalogue is consistent across version sites. Package version,
content revision, presentation revision, Askr revision and branch remain
separate fields in generated build records.

The Documentation Pages workflow builds on develop/main pushes or manual
workflow dispatch; pull requests do not publish. A required render failure
prevents publication and leaves the last deployed site in place. Source changes
follow develop-to-main promotion; stable package metadata belongs to main.
Version/catalogue changes should be checked with the existing maintained
version tooling and the delivery packet's assigned observations.

## Implementation and project references

- [Architecture and data flow](docs/architecture.qmd)
- [Runtime and result provenance](docs/provenance.qmd)
- [Project scope and derivation](docs/goals-and-upstream.qmd)
- [Project direction](docs/roadmap.qmd)
- [Historical diagnostic archive](docs/errors.qmd)

The integrated derivation inputs were gsim downstream
`0.2.0+scq.1`, revision `8f5dc6c05255d003a9c6d8959537bcf8068379d3`,
and historical SGB revision `e74a343154c6b19b6ba32d6fb297e700cfe08ff2`.
They explain origin, not runtime dependencies, support expansion or API
compatibility. Current source geometry uses the in-tree `scgsim.geometry` API
and `scgsim.geometry.compiler` for route planning and derived construction.

## Engineering convergence

The current engineering-convergence work remains `CONVERGING`. Track semantic
state and repository delivery separately in
[the convergence status page](docs/convergence-status.qmd).

The required delivery route is a pull request with required checks and no
bypass. The literal `required` check always reaches a terminal result and
accepts only the results declared for the selected change scope. Failed or
cancelled jobs and unexpected skips remain failures.

CI first classifies the exact event comparison. Draft pull requests run this
classification and intentionally skip product jobs; marking a pull request
Ready starts the applicable checks. Proven Markdown prose changes in
`README.md` or `docs/**/*.md` receive local Markdown link-path validation.
Documents with inline-code ambiguity stay on the product path. Executable or
unsupported content, QMD/Notebook files, shared settings, dependencies and source
changes also use the product checks. An unavailable comparison also uses
the product checks.

Non-Draft product changes and manual dispatch retain the Python 3.12/3.13
package-version, distribution-build, editable-import and isolated-wheel-import
checks. The Python 3.12 offline job runs the full public offline test suite,
strict Pyright on the three configured modules and Ruff F on tracked Python
files. The separate Python 3.12 `docs-smoke` job runs current source examples
through file-only preparation with the pinned public PDK. CI uses public
inputs and excludes private golden data; it does not invoke AEDT or Palace
solvers.

Pages renders the frozen version catalogue, stable pages and history on
`develop` and `main` pushes or manual dispatch, with code execution disabled.
It remains separate because this coverage differs from file-only preparation;
it is not a pull-request check.

For each fix pull request, include a regression case or explain why one is not
included, the alternative evidence used, and any remaining gap. For calculation
changes, use independent public analytic regressions to establish normalization,
participation, and group identities, and report them separately from returned-data
compatibility. A historical aggregate whose calculation procedure is unknown is
not a correctness oracle; preserve it as history and do not derive a tolerance
from it.

For an authorized real-data compatibility check, report only permitted artifact
hashes, identity, and receipt references. State whether raw records, including
unknown values, are preserved through the public result, and report missing-input,
content-mismatch, and identity-mismatch negatives. Keep private values, names,
paths, and payloads out of public text. If the check has not finished, mark it
`IN PROGRESS`.

For an affected Tier A change, report the assigned private golden status. If
required data or access are unavailable, record `NOT RUN` and leave the
associated claim open. An owner-authorized run uses the public test entry point
with an explicit run flag and manifest:

```sh
python -m pytest tests/test_private_golden.py \
  --run-private-golden \
  --private-golden-manifest /path/to/authorized-manifest.json
```

The manifest is an input to the reader, not permission to access or publish
its data. Missing required manifest input is a usage error; a skipped test is
`NOT RUN`, not a golden result. Public CI excludes the `private_golden` marker.
Keep golden data outside the public repository. Tier B remains supported and
is not reclassified by this convergence work. Use the
[pull request template](.github/PULL_REQUEST_TEMPLATE.md) and the
[convergence status page](docs/convergence-status.qmd) for current scope and
open items.

## Larger proposals across projects

Send a major architecture, cross-package contract, or system-wide proposal to
the nearest owning Lead. Before writing an execution plan, that Lead checks the
repository, branch and base revision, related source, open work, contracts, and
existing capabilities. Keep the plan in the current work packet. For authorized
delivery, the repository's Git owner completes the assigned branch and push
steps and reads back the remote result; a proposal does not transfer that
ownership.

## Historical structural acceptance

The v2 module and API structure was accepted on 2026-10-09 and later integrated
through PR34. That historical milestone describes the structure only; it does
not set the semantic or delivery state of the separate engineering-convergence
work.
