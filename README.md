---
title: "SCGSim"
output-file: index.html
---

<span id="scgsim"></span>SCGSim prepares geometry-based inputs and backend-specific solver handoffs for
superconducting-circuit simulations, then resolves and presents returned
results. It is an independent downstream research toolkit.

## Choose a workflow {#documentation-source}

Start with the [Examples and tutorials](docs/examples.qmd) page. It builds the
shared Xmon inputs first, then points to the Palace or AEDT workflow.

### Follow the tutorial path {#workflows}

1. [Build the shared Xmon inputs](docs/tutorial-xmon-input.qmd).
2. Choose the [Palace handoff](docs/tutorial-xmon-palace.qmd) or the
   [AEDT handoff and EPR workflow](docs/tutorial-xmon-aedt.qmd).
3. Read [Eigenmode methods and offline EPR](docs/eigenmode-epr.qmd) for
   normalization, loss assumptions, and returned-result analysis.

The backend pages document separate preparation and result APIs. Their static
examples do not claim a native geometry check or solver run.

## Optional installation extras

SCGSim supports Python 3.12 and 3.13. Install ordinary AEDT workflows with
`pip install 'scgsim[aedt]'`; this extra includes GDS support and PyAEDT 1.3.0.
For AEDT Surface-EPR preparation, install `pip install 'scgsim[aedt-epr]'`;
it adds the supported GDSFactory 9.x and KLayout 0.30.x ranges while retaining
the AEDT requirements. Python 3.13 package support alone does not establish
native AEDT 2024.2 correctness; see the [AEDT runtime
reference](docs/specs/aedt-runtime.qmd).

When SCGSim and OrPen are used in one interpreter, resolve both packages
together so the consumer project's lock selects one compatible GDSFactory,
kfactory, and KLayout set. The [OrPen Xmon tutorials](docs/examples.qmd) link
to the public notebook sources they use.

## Online documentation

The [SCGSim documentation home](https://orpenstrike.github.io/scgsim/) lists
published versions. Open the [development documentation](https://orpenstrike.github.io/scgsim/develop/)
for the current `develop` site.

<span id="reference"></span>

<span id="current-nonclaims"></span>

## More information

### Package capabilities and limits

The [Backend Support Matrix](docs/backend-support.qmd) distinguishes implemented
capabilities, exercised evidence, and current limits. Palace Driven and
Magnetostatic workflows are not implemented.

### Project context {#project-context}

The [project goals and upstream relationship](docs/goals-and-upstream.qmd) and
[architecture guide](docs/architecture.qmd) describe package ownership,
source boundaries, and fallback limits. See [Provenance and data
boundaries](docs/provenance.qmd) for evidence and publication authority.
