# SCGSim Agent Guide

`OrPenStrike/scgsim` is SCQ_Design's sole current reusable solver, runtime,
and result-production authority. The SCGSim Development Lead owns product
source in this repository. The SCGSim Documentation Lead owns reader-facing
documentation only for paths explicitly assigned by the Human.

The Human may work directly with the Documentation Lead on documentation.
That Lead owns information architecture, narrative, navigation, terminology,
and writing for assigned README, documentation, and runbook paths. The
Development Lead retains runtime, API, schema, formula, scientific-meaning,
docstring, generator, packaged-knowledge program behavior, executable
tutorial/notebook code, test, CI, packaging, and release ownership. Mixed
prose and executable examples, notebooks, or shared configuration need exact
path and hunk allocation with one author; prose cannot change product or
research contracts. The Development and Integration Leads retain serialized
Git ownership under `develop-main`.

After a coherent documentation candidate or checkpoint, the Documentation Lead
directly notifies the corresponding Development Lead with its exact source
identity and implementation-facing changes. The Development Lead reconciles
any actual discrepancy; a documentation change alone does not authorize
changes to product semantics, new tests, or solver runs. The Documentation
Lead may own the existing Preview operation after its owner explicitly
transfers it. Keep the running preview in place: a transfer does not authorize
restarting or duplicating it, or transfer public publishing authority.

Use the `develop-main` delivery profile. Route collaboration,
lifecycle, ownership, and delivery policy through `$scq-collaboration-roles`,
which composes the V1 lifecycle and model-routing policies. This adapter does
not duplicate those policies.

Read first: `README.md`, `docs/goals-and-upstream.qmd`,
`docs/architecture.qmd`, `docs/geometry-sgb.qmd`, and `docs/ownership.qmd`.
Then read the smallest relevant technical contract before editing.

External gsim and historical SGB are derivation provenance only: new SCGSim
work must use the in-tree `scgsim.geometry` source APIs and the advanced
`scgsim.geometry.compiler` APIs, never a direct external consumer or fallback.
OrPen SC PDK owns public component-simulation notebooks;
SCGSim owns no duplicate notebook source.
