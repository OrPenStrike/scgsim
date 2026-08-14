---
title: "SCGSim"
output-file: index.html
---

# SCGSim

SCGSim is an independent downstream research toolkit for superconducting-circuit
simulation workflows. It prioritizes reproducible research behavior and stable
consumer contracts. It is **not official gsim**, does not erase or replace
upstream work, and does not promise Human review of Agent-driven code changes.

The integrated Phase-1 scaffold is deliberately small: an
installable/importable `scgsim` package, Python `~=3.12.0`, version
`0.1.0.dev0`, and build/CI packaging checks. It has no runtime dependencies.

No Palace or AEDT backend, solver, report, execution, handoff, or notebook API
is implemented.

Planned public namespaces are `scgsim.palace` and `scgsim.aedt`. They are names
for future contracts, not reachable Phase-1 APIs.

## Read the site

- [Goals and upstream relationship](docs/goals-and-upstream.qmd)
- [Examples](docs/examples.qmd)
- [Architecture and data flow](docs/architecture.qmd)
- [Backend support matrix](docs/backend-support.qmd)
- [Notebook UX contracts](docs/notebook-ux.qmd)
- [Roadmap and current status](docs/roadmap.qmd)

## Current nonclaims

There is no live Palace solve, numerical Surface-EPR result, backend support,
cloud fallback, release, deployment, or published private evidence in this
repository state.
