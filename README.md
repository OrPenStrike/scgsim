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

The converging geometry candidate starts from component-local entities and
named occurrences. A Notebook-authored `GeometryPlan` applies the final
`Net -> Entity` map and prepares one immutable input for the Palace and AEDT
backends. The [GeometryPlan and SGB guide](docs/geometry-sgb.qmd) documents
this source-identity contract.

The current `CONVERGING` package provides the in-tree `scgsim.sgb` Core,
Palace Electrostatic/Eigenmode geometry-to-report workflows, and version-locked
AEDT handoff/run/resolve workflows for HFSS Driven Terminal/Modal, HFSS
Eigenmode, Q3D, and Q2D. The Eigenmode candidate includes backend-specific
surface, bulk, and port/junction EPR analysis, offline film reanalysis, and
owner-and-coverage reports. These are implemented candidates, not V1-stable
contracts; the two backends retain separate preparation and result APIs. The
[Backend Support Matrix](docs/backend-support.qmd) separates implementation
status from exercised evidence and current limits. Static EPR examples do not
claim native solver validation.

The package supports Python 3.12 and 3.13. The optional AEDT integration pins
PyAEDT 1.3.0; Python 3.13 package support does not by itself establish native
AEDT 2024.2 operation correctness.

## Optional installation extras

Install the ordinary AEDT workflows with `pip install 'scgsim[aedt]'`. The
`aedt` extra includes GDS support and PyAEDT 1.3.0, but does not install the
additional layout-geometry dependencies used by AEDT Surface-EPR preparation.
For that path, install `pip install 'scgsim[aedt-epr]'`; this extra adds the
supported GDSFactory 9.x and KLayout 0.30.x ranges while retaining the same
AEDT requirements.

When SCGSim and OrPen are used in one interpreter, resolve both packages
together so the consumer project's lock selects one compatible GDSFactory,
kfactory, and KLayout set. SCGSim's development `uv.lock` resolves SCGSim's own
extras; it is not a cross-project pin for an OrPen consumer environment.

OrPen SC PDK owns the public component-simulation notebooks. SCGSim owns no
duplicate notebook source and has no runtime dependency on OrPen.

## Read the site

### Workflows

- [Choose a workflow and find public notebooks](docs/examples.qmd)
- [Build the shared public Xmon inputs](docs/tutorial-xmon-input.qmd)
- [Prepare a Palace Route B Xmon Eigenmode handoff](docs/tutorial-xmon-palace.qmd)
- [AEDT Route B Xmon Eigenmode and EPR guide](docs/tutorial-xmon-aedt.qmd)
- [Eigenmode methods, EPR, and offline reanalysis](docs/eigenmode-epr.qmd)
- [Notebook UX contracts](docs/notebook-ux.qmd)
- [Execution profiles and handoff](docs/execution-profiles.qmd)

### Reference

- [Backend implementation, evidence, and limits](docs/backend-support.qmd)
- [AEDT runtime and data contracts](docs/specs/aedt-runtime.qmd)
- [Palace Electrostatic and SGB contracts](docs/specs/palace-electrostatic-sgb.qmd)
- [Unified report model](docs/report-model.qmd)
- [Architecture and data flow](docs/architecture.qmd)

### Project context

- [Goals and upstream relationship](docs/goals-and-upstream.qmd)
- [Roadmap](docs/roadmap.qmd)
- [Provenance and data boundaries](docs/provenance.qmd)
- [Error archive](docs/errors.qmd)
- [Folder tree and ownership](docs/ownership.qmd)

## Current nonclaims

Palace Driven and Magnetostatic are not implemented; the [support matrix](docs/backend-support.qmd)
has the complete capability state. SCGSim does not use cloud fallback or grant
publication authority for private evidence.
