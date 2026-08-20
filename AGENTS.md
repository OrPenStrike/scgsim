# SCGSim Agent Guide

`OrPenStrike/scgsim` is SCQ_Design's sole current reusable solver, runtime,
and result-production authority. The SCGSim Development Lead is the sole owner
of tracked source in this repository.

Work only in the `scgsim-physical-checkout` physical checkout
`/home/ili/Githubs/SCQ_Design/scgsim`, targeting `develop`; do not create an
SCGSim worktree. Route collaboration, lifecycle, ownership, and delivery policy through
`$scq-collaboration-roles`, which composes the V1 lifecycle and model-routing
policies. This adapter does not duplicate those policies.

Read first: `README.md`, `docs/goals-and-upstream.qmd`,
`docs/architecture.qmd`, `docs/geometry-sgb.qmd`, and `docs/ownership.qmd`.
Then read the smallest relevant technical contract before editing.

External gsim and historical SGB are derivation provenance only: new SCGSim
work must use the in-tree `scgsim.sgb` authority, never a direct external
consumer or fallback. OrPen SC PDK owns public component-simulation notebooks;
SCGSim owns no duplicate notebook source.
