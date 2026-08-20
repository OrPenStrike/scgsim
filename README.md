---
title: "SCGSim"
output-file: index.html
---

# SCGSim

SCGSim is an independent downstream research toolkit for superconducting-circuit
simulation workflows. It prioritizes reproducible research behavior and stable
consumer contracts. It is **not official gsim**, does not erase or replace
upstream work, and does not promise Human review of Agent-driven code changes.

Within SCQ_Design, SCGSim is the sole current reusable solver, runtime, and
result-production authority. External gsim and historical SGB remain derivation
provenance only: new SCGSim work must not consume them directly or use them as a
fallback; the in-tree `scgsim.sgb` Core is the current geometry and topology
authority.

The current `CONVERGING` package provides the in-tree `scgsim.sgb` Core,
Palace Electrostatic/Eigenmode geometry-to-report workflows, and version-locked
AEDT handoff/run/resolve workflows for HFSS Driven Terminal/Modal, HFSS
Eigenmode, Q3D, and Q2D. These are implemented candidates, not V1-stable
contracts.

OrPen SC PDK owns the public component-simulation notebooks. SCGSim owns no
duplicate notebook source and has no runtime dependency on OrPen.

## Read the site

- [Goals and upstream relationship](docs/goals-and-upstream.qmd)
- [Examples](docs/examples.qmd)
- [Architecture and data flow](docs/architecture.qmd)
- [Backend support matrix](docs/backend-support.qmd)
- [Notebook UX contracts](docs/notebook-ux.qmd)
- [Roadmap and current status](docs/roadmap.qmd)

## Current nonclaims

Palace Driven and Magnetostatic remain unimplemented. There is no cloud
fallback, release, deployment, or publication authority for private evidence.
