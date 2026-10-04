# SCGSim

SCGSim turns superconducting-circuit source geometry into backend inputs,
explicit solver handoffs and verified returned results. It is an independent
downstream research toolkit. The in-tree `scgsim.sgb` owns geometry lowering;
`scgsim.palace` and `scgsim.aedt` own their solver workflows and physical result
meaning.

## Learn the workflow

Start with the [simulation course](docs/examples.qmd). It takes you from
[installation](docs/course/installation.qmd) through source observations,
backend configuration, actual host execution, complete result retrieval and
offline interpretation. Palace Electrostatic/Eigenmode and AEDT
Eigenmode/EPR, Driven Modal/Terminal, Q3D and Q2D each have a dedicated branch.

Public component-simulation notebooks belong to OrPen SC PDK; the course links
their canonical sources rather than maintaining another notebook copy. The
site displays code without running kernels or solvers. Prepared files,
native geometry observations and returned field results are distinct evidence.

## Capabilities and interpretation

The [backend support matrix](docs/backend-support.qmd) records implemented
candidate workflows and their current evidence limits. Broad V1 semantics
remain **CONVERGING**. Palace Driven and Magnetostatic are not implemented;
Meep is excluded.

Read [geometry concepts](docs/concepts/geometry.qmd) to understand source
Entities, occurrences and final Nets, and [EPR concepts](docs/eigenmode-epr.qmd)
to interpret participation, loss assumptions and uncomputed channels. Exact
behavior belongs to the linked contracts; [architecture](docs/architecture.qmd)
explains implementation responsibilities.

## Documentation and ownership

The [documentation home](https://orpenstrike.github.io/scgsim/) lists published
versions. The [development site](https://orpenstrike.github.io/scgsim/develop/)
records its source revision and package version. Use that identity with the
matched public input revision; a development version string alone does not
identify every API change.

SCGSim owns runtime/package/documentation source. OrPen owns public PDK facts,
components and component notebooks. [Ownership](docs/ownership.qmd) describes
delivery responsibility; [provenance](docs/provenance.qmd) defines public/private
data boundaries; [goals and derivation history](docs/goals-and-upstream.qmd)
distinguish current authority from upstream provenance.
