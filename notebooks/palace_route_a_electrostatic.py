# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Route A Electrostatic Candidate

# %% [markdown]
# ## Design And Geometry Controls
# Public synthetic GDS and an explicit SGB stack; Route A uses the same component/stack path.

# %%
from pathlib import Path

GEOMETRY_CONTROLS = {
    "route": "A",
    "terminal_name": "public_terminal",
    "terminal_net": "PUBLIC_NET",
}

# %% [markdown]
# ## Meshing Controls

# %%
MESH_CONTROLS = {"refined_mesh_size": 25.0, "max_mesh_size": 80.0}

# %% [markdown]
# ## Solver Controls

# %%
SOLVER_CONTROLS = {
    "order": 1,
    "tolerance": 1e-6,
    "max_iterations": 400,
    "solver_type": "Default",
    "preconditioner": "Default",
    "device": "CPU",
}

# %% [markdown]
# ## Execution Controls

# %%
EXECUTION_CONTROLS = {
    "profile_names": ("ltlab-local", "ltlab-slurm", "f1-slurm"),
    "selected_profile": "ltlab-slurm",
    "executable": "palace",
    "setup_commands": ("module load palace",),
    "resources": {
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 1,
        "command_style": "binary",
    },
}

# %% [markdown]
# ## Output And Run Identity Controls

# %%
OUTPUT_CONTROLS = {"run_dir": Path(".artifacts") / "public_route_a_electrostatic"}

# %% [markdown]
# ## Validation And Failure Controls

# %%
VALIDATION_CONTROLS = {
    "msh_version": "2.2",
    "terminal_count": 1,
    "surface_epr_types": ("MA", "MS", "SA"),
}

# %% [markdown]
# ## Data Classification And Provenance
# Runtime authority is `scgsim.sgb`. SGB base/import SHAs are derivation history;
# gsim mesh methodology is `8f5dc6c05255d003a9c6d8959537bcf8068379d3`.

# %%
PROVENANCE = {
    "data_classification": "public-synthetic",
    "scgsim_sgb": "scgsim.sgb",
    "sgb_derivation_base": "e74a343154c6b19b6ba32d6fb297e700cfe08ff2",
    "sgb_derivation_imported": "f3fd898d6e4eaf31595c9aaca6a0658f0cb7f3b1",
    "gsim_mesh_methodology": "8f5dc6c05255d003a9c6d8959537bcf8068379d3",
    "gsim_portable_handoff": "8cf5fa79fa3abb176940dbfc520ff34a44f4770e",
    "palace_runtime": "0.16.1",
    "palace_schema": "0.16.0",
}

# %%
import json

import gdsfactory as gf

from scgsim.palace import ElectrostaticSim

EPR_SPECS = {
    "MA": {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0},
    "MS": {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0},
    "SA": {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0},
}

# %% [markdown]
# ## Build Component

# %%
gf.clear_cache()
gf.gpdk.PDK.activate()
component = gf.Component("public_route_a_plate")
component.add_polygon([(0, 0), (80, 0), (80, 80), (0, 80)], layer=(1, 0))
stack = {
    "solution_regions": {
        "AIR_ABOVE": {
            "role": "solution_region",
            "is_airbox": True,
            "material_id": "vacuum",
            "geometry_kind": "domain",
            "geometry": {
                "domain": "AIR_ABOVE",
                "z_min_um": 0,
                "z_max_um": 40,
                "padding_um": 50,
            },
        },
        "SUBSTRATE": {
            "role": "solution_region",
            "material_id": "substrate",
            "geometry_kind": "domain",
            "geometry": {
                "domain": "SUBSTRATE",
                "z_min_um": -20,
                "z_max_um": 0,
                "padding_um": 50,
            },
        },
    },
    "materials": {
        "vacuum": {"kind": "vacuum", "permittivity": 1.0, "loss_tangent": 0.0},
        "substrate": {"kind": "dielectric", "permittivity": 11.45, "loss_tangent": 0.0},
        "aluminum": {"kind": "conductor"},
    },
    "layers": [
        {
            "layer": 1,
            "datatype": 0,
            "semantic_id": "PUBLIC_METAL",
            "role": "metal",
            "material_id": "aluminum",
            "priority": 1,
            "part_role": "face_metal",
            "net_id": "PUBLIC_NET",
            "geometry_kind": "layout_extrusion",
            "host_void_semantic_id": "AIR_ABOVE",
            "geometry": {
                "z_um": 0,
                "thickness_um": 0.2,
                "geometry_source": "gds_polygon",
            },
            "route_representations": {
                "A": "surface_sheet",
                "B": "cutout_boundary_shell",
            },
            "metadata": {"source_layer_name": "PUBLIC_METAL"},
        }
    ],
}

# %% [markdown]
# ## Configure Problem And EPR

# %%
sim = ElectrostaticSim()
sim.set_geometry(component)
sim.set_stack(stack)
sim.set_output_dir(OUTPUT_CONTROLS["run_dir"])
sim.set_airbox(margin_x=25, margin_y=25, z_above=20)
sim.set_surface_epr(representation=GEOMETRY_CONTROLS["route"], specs=EPR_SPECS)
sim.add_terminal(
    GEOMETRY_CONTROLS["terminal_name"], net_id=GEOMETRY_CONTROLS["terminal_net"]
)
sim.set_electrostatic(
    save_fields=0,
    unassigned_conductor_policy="error",
    exterior_boundary_policy="ground",
)
sim.set_numerical(**MESH_CONTROLS, **SOLVER_CONTROLS)

# %% [markdown]
# ## Build Mesh

# %%
run_dir = OUTPUT_CONTROLS["run_dir"]
mesh_path = sim.mesh()
assert mesh_path.name == "palace.msh"
assert f"$MeshFormat\n{VALIDATION_CONTROLS['msh_version']} 0 8" in mesh_path.read_text(
    encoding="utf-8"
)
manifest = json.loads(
    (run_dir / "metadata" / "mesh_manifest.json").read_text(encoding="utf-8")
)
assert any(group["section"] == "volumes" for group in manifest["groups"])

# %% [markdown]
# ## Write And Validate Config

# %%
config_path = sim.write_config()
config = json.loads(config_path.read_text(encoding="utf-8"))
assert "Metadata" not in config
assert config["Boundaries"]["Postprocessing"]["Dielectric"]
index_map = json.loads(
    (run_dir / "metadata" / "palace_index_map.json").read_text(encoding="utf-8")
)
terminal_entries = [
    entry for entry in index_map["entries"] if entry["section"] == "Boundaries.Terminal"
]
assert len(terminal_entries) == VALIDATION_CONTROLS["terminal_count"]
terminal_entry = terminal_entries[0]
assert terminal_entry["terminal_name"] == GEOMETRY_CONTROLS["terminal_name"]
assert terminal_entry["net_id"] == GEOMETRY_CONTROLS["terminal_net"]
terminal_groups = [
    group
    for group in manifest["groups"]
    if group["section"] == "boundary_surfaces"
    and group["structured"]
    and group["name"] in terminal_entry["physical_names"]
]
assert terminal_groups
assert {group["conductor_component_id"] for group in terminal_groups} == set(
    terminal_entry["conductor_component_ids"]
)
assert all(
    group["net_id"] == GEOMETRY_CONTROLS["terminal_net"] for group in terminal_groups
)
epr_types = {
    entry["metadata"]["interface_type"]
    for entry in index_map["entries"]
    if entry["section"] == "Boundaries.Postprocessing.Dielectric"
}
assert epr_types == set(VALIDATION_CONTROLS["surface_epr_types"])

# %% [markdown]
# ## Prepare And Inspect Handoff

# %%
plan = sim.prepare_handoff(
    profile=EXECUTION_CONTROLS["selected_profile"],
    executable=EXECUTION_CONTROLS["executable"],
    resources=EXECUTION_CONTROLS["resources"],
    setup_commands=EXECUTION_CONTROLS["setup_commands"],
)
assert plan.script_path.name == "run_palace.sbatch" and plan.archive_path.is_file()
handoff = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
assert handoff["status"] == "prepared" and handoff["problem"] == "Electrostatic"
assert handoff["source_revisions"]["scgsim_sgb"] == PROVENANCE["scgsim_sgb"]
assert (
    handoff["source_revisions"]["sgb_derivation"]["base"]
    == PROVENANCE["sgb_derivation_base"]
)
assert (
    handoff["source_revisions"]["sgb_derivation"]["imported_development"]
    == PROVENANCE["sgb_derivation_imported"]
)
assert (
    handoff["source_revisions"]["gsim_meshing"] == PROVENANCE["gsim_mesh_methodology"]
)
assert (
    handoff["source_revisions"]["gsim_portable_handoff"]
    == PROVENANCE["gsim_portable_handoff"]
)
assert (
    handoff["palace_identity"]["runtime"].removeprefix("v")
    == PROVENANCE["palace_runtime"]
)
assert (
    handoff["palace_identity"]["config_schema"].removeprefix("v")
    == PROVENANCE["palace_schema"]
)

# %% [markdown]
# ## Physics Analysis Results
# Deferred: this notebook prepares a manual handoff and does not solve.

# %% [markdown]
# ## Simulation Performance / Benchmarks
# Palace was not run; solver cost is unavailable. The following values describe
# real preparation artifacts only.

# %%
preparation_facts = {
    "mesh_bytes": mesh_path.stat().st_size,
    "mesh_groups": len(manifest["groups"]),
    "handoff_archive_bytes": plan.archive_path.stat().st_size,
    "data_classification": PROVENANCE["data_classification"],
    "runtime_authority": handoff["source_revisions"]["scgsim_sgb"],
    "solver_cost": "unavailable: Palace was not run",
}
preparation_facts  # noqa: B018
