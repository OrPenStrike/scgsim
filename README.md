---
title: "SCGSim"
output-file: index.html
---

# SCGSim

SCGSim is an independent downstream research toolkit for superconducting-circuit
simulation workflows. It prioritizes reproducible research behavior and stable
consumer contracts. It is **not official gsim**, does not erase or replace
upstream work, and does not promise Human review of Agent-driven code changes.

The Phase-1 candidate is deliberately small. This documentation candidate and
a separate runtime candidate are both **NOT_INTEGRATED**. The exact `develop`
base remains empty, while the runtime candidate implements only an
installable/importable `scgsim` package, Python `~=3.12.0`, version
`0.1.0.dev0`, and build/CI packaging checks. It adds no runtime dependencies.

No Palace or AEDT backend, solver, report, execution, handoff, or notebook API
is implemented in either candidate.

Planned public namespaces are `scgsim.palace` and `scgsim.aedt`. They are names
for future contracts, not reachable Phase-1 APIs.

## Read the site

- [Goals and upstream relationship](docs/goals-and-upstream.qmd)
- [Architecture and data flow](docs/architecture.qmd)
- [Backend support matrix](docs/backend-support.qmd)
- [Notebook UX contracts](docs/notebook-ux.qmd)
- [Roadmap and current status](docs/roadmap.qmd)

## Current nonclaims

There is no live Palace solve, numerical Surface-EPR result, backend support,
cloud fallback, release, deployment, or published private evidence in this
repository state.
