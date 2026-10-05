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

Use the returned quantity and its recorded history to answer your stated SCQ
research question. A native convergence flag records the outcome of your
configured stopping criterion; it does not decide whether the result is adequate
for that use. The [result-reading lesson](docs/course/returned-results.qmd#interpret-for-scq-use)
shows how to retain criteria, changes, units and native status while making
that research judgement.

## Capabilities and interpretation

The [backend support matrix](docs/backend-support.qmd) records accepted V1
workflows and their current evidence limits. The Human accepted
the implemented V1 semantics, and the reviewed assigned stabilization is
complete. Stable `main` 1.0.0 is published; delivery identities and native evidence
remain separate. Palace Driven and Magnetostatic are not implemented;
Meep is excluded.

Read [geometry concepts](docs/concepts/geometry.qmd) to understand source
Entities, occurrences and final Nets, and [EPR concepts](docs/eigenmode-epr.qmd)
to interpret participation, loss assumptions and uncomputed channels. Exact
behavior belongs to the linked contracts; [architecture](docs/architecture.qmd)
explains implementation responsibilities.

## Documentation and ownership

The four documentation Areas are **Overview**, **Tutorial**, **Concept**, and
**Contract**. Each Area's **Pages** menu selects a document; **Sections** on the
right navigates within it. Implementation and project pages live under Contract rather
than a separate top-level Area.

Stable `main` carries the latest stable release; `develop` contains the next
prerelease line. Native Askr presentation is confirmed for the 1.0.1
maintenance release. Instructional content revisions remain separate work;
this release does not reopen the stabilized runtime V1 contract.

The [documentation home](https://orpenstrike.github.io/scgsim/) selects the
latest published stable version. Versioned sites keep a matching package
version and content revision; the version menu distinguishes development,
current stable, and retained history. The `/main/` and `/develop/` links remain
aliases for the corresponding selected versions. A historical page keeps its
original content even when its presentation is updated. Use its recorded
content identity with the matched public input revision; a version string
alone does not identify every API change.

SCGSim owns runtime/package/documentation source. OrPen owns public PDK facts,
components and component notebooks. [Ownership](docs/ownership.qmd) describes
delivery responsibility; [provenance](docs/provenance.qmd) defines public/private
data boundaries; [goals and derivation history](docs/goals-and-upstream.qmd)
distinguish current authority from upstream provenance.
