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

## SCGSim 2.0 structure candidate

The structural/API redesign is `CONVERGING`. The frozen intake baseline is
`develop@3a52fb1`, tree `96adcf8672b497a29dd0a6b43043cca11efc3043`. The
current source tree contains the v2 geometry, Palace, AEDT, visualization, and
shared-presentation ownership layout; these changes are not yet integrated or
delivered. Consumers keep their existing pins.

The responsibility tree, ownership rules and baseline-to-current public
submodule/symbol inventory are recorded in
[API migration](docs/api-migration.qmd). The candidate is still converging and
source implementation remains under review. Implement one authority per accepted
responsibility, without old `scgsim.sgb` compatibility wrappers, revived Q3D
input-v1/v2, or a common solver `Simulation`/physical-result abstraction. Keep
`python -m scgsim.aedt.run` as the supported AEDT transaction CLI. The current
v2 AEDT producer records runtime-source v13 with 61 canonical declared module
paths, including 41 AEDT modules. Its measured module bytes and content digest
are required; truthful Git revision observation may be unavailable. V13 does
not include a helper-hash map or claim transitive-source coverage. Readers
retain the exact v1–v12 historical path and identity rules. Existing Q3D
body-v3 native consumer behavior is still unobserved and must not be described
as completed by the structure plan.

The Development Lead retains tracked-source and integration ownership. Fixed
workers use disjoint, time-bounded path leases. After Human acceptance of the
new API, Human Plan §4 authorizes maintaining only affected existing tests'
import and patch locations and running those affected tests. Preserve test
expectations; this design phase adds or changes no test behavior. Dependency
pins, consumer environments and release metadata remain unchanged until
separately assigned.
