# SCGSim Agent Guide

`NCUPHY-LTLab/scgsim` is SCQ_Design's sole current reusable solver, runtime,
and result-production authority. SCGSim Documentation independently owns
README files, guides, explanations of established APIs, document code examples,
site navigation, links, anchors, cards, CSS/theme adoption, and
documentation-only configuration or rendering helpers through its own workers.
Development owns product/runtime/schema behavior and scientific/model meaning.

After protecting active writers through the one-time handoff, ordinary
documentation work needs no new Development release or prose approval.
Coordinate only actual overlapping hunks. Notify Development for
synchronization and send the exact candidate to the original Git owner for
commit, push, and remote readback; this delivery does not transfer authoring.
Documentation ownership does not authorize new solver or scientific execution.

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
