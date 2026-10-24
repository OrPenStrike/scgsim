# SCGSim Agent Guide

`NCUPHY-LTLab/scgsim` is SCQ_Design's reusable layout, solver, runtime,
and result-production authority. The SCGSim repository Lead owns product
source, documentation, the documentation site and Preview operations.

Use the `develop-main` delivery profile and `$scq-collaboration-roles` for
ownership and delivery; it composes the lifecycle and model-routing policies.
The existing Git owner serializes commits and pushes. Integration retains
protected merge authority.

The Lead may make small, tightly coupled edits directly. Delegate independent
work to the existing team when useful, including a documentation writer for
substantial writing. Assign one writer per path or exact hunk and preserve
other writers' WIP and running services. Routine documentation needs no
separate Lead approval; unresolved product or scientific meaning returns to
the repository Lead.

Read the task-relevant entry and smallest applicable contract. `README.md`,
`docs/goals-and-upstream.qmd`, `docs/architecture.qmd`, `docs/geometry-sgb.qmd`
and `docs/ownership.qmd` are reference entries, not a mandatory reading batch.
Authorized implementation includes directly related documentation, tests and
local checks under `$v1-lifecycle`; research and native execution follow the
assigned task scope.

External gsim and historical SGB are derivation provenance only: new work uses
in-tree `scgsim.geometry` and advanced `scgsim.geometry.compiler` APIs, never
an external consumer or fallback. OrcaPen SC PDK owns public component-simulation
notebooks; SCGSim owns no duplicate notebook source.
