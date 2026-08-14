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
# # Route B Eigenmode Candidate

# %% [markdown]
# ## Design And Geometry Controls
# Public synthetic rounded arms share one logical face-metal layer; the authored
# port-sheet polygon crosses only their local gap.

# %%
from pathlib import Path

GEOMETRY_CONTROLS = {
    "route": "B",
    "logical_metal": "PUBLIC_ARM_METAL",
    "port_name": "gap_sheet",
    "port_layer": (2, 0),
    "port_direction": [1.0, 0.0, 0.0],
    "port_inductance_h": 1e-12,
}

# %% [markdown]
# ## Meshing Controls

# %%
MESH_CONTROLS = {"refined_mesh_size": 12.0, "max_mesh_size": 50.0}

# %% [markdown]
# ## Solver Controls

# %%
SOLVER_CONTROLS = {
    "num_modes": 2,
    "target": 5e9,
    "tolerance": 1e-6,
    "save": 0,
    "order": 1,
    "max_iterations": 400,
    "solver_type": "Default",
    "preconditioner": "Default",
    "device": "CPU",
}

# %% [markdown]
# ## Execution Controls
# The three profile names describe manual execution intent. This notebook prepares
# one LTlab single-node Slurm handoff only and never calls Palace or submits work.

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
OUTPUT_CONTROLS = {"run_dir": Path(".artifacts") / "public_route_b_eigenmode"}

# %% [markdown]
# ## Validation And Failure Controls

# %%
VALIDATION_CONTROLS = {
    "msh_version": "2.2",
    "port_owner_count": 2,
    "embedded_solution_volume_count": 1,
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
    "palace_runtime": "0.16.1",
    "palace_schema": "0.16.0",
}

# %%
import json
import math

import gdsfactory as gf

from scgsim.palace import EigenmodeSim

EPR_SPECS = {
    "MA": {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0},
    "SA": {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0},
}
# This stack places the face metal entirely in AIR_ABOVE. After meshing, the
# structured manifest below proves it has no MS interface records, so no
# required MS EPR spec is fabricated for this Route B example.

# %% [markdown]
# ## Build Component

# %%
gf.clear_cache()
gf.gpdk.PDK.activate()
component = gf.Component("public_route_b_rounded_arm_gap")


def capsule(x0: float, x1: float, radius: float) -> list[tuple[float, float]]:
    left = [
        (x0 + radius * math.cos(angle), radius * math.sin(angle))
        for angle in [math.pi / 2 + index * math.pi / 16 for index in range(17)]
    ]
    right = [
        (x1 - radius + radius * math.cos(angle), radius * math.sin(angle))
        for angle in [-math.pi / 2 + index * math.pi / 16 for index in range(17)]
    ]
    return [*left, *right]


component.add_polygon(capsule(-60, -6, 6), layer=(1, 0))
component.add_polygon(capsule(6, 60, 6), layer=(1, 0))
component.add_polygon([(-10, -4), (10, -4), (10, 4), (-10, 4)], layer=(2, 0))
component.add_port(
    name=GEOMETRY_CONTROLS["port_name"],
    center=(0, 0),
    width=8,
    orientation=0,
    layer=GEOMETRY_CONTROLS["port_layer"],
)
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
                "z_max_um": 35,
                "padding_um": 35,
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
                "padding_um": 35,
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
            "semantic_id": GEOMETRY_CONTROLS["logical_metal"],
            "role": "metal",
            "material_id": "aluminum",
            "priority": 1,
            "part_role": "face_metal",
            "net_id": "PUBLIC_ARM_NET",
            "geometry_kind": "layout_extrusion",
            "host_void_semantic_id": "AIR_ABOVE",
            "geometry": {
                "z_um": 1,
                "thickness_um": 0.2,
                "geometry_source": "gds_polygon",
            },
            "route_representations": {
                "A": "surface_sheet",
                "B": "cutout_boundary_shell",
            },
            "metadata": {"source_layer_name": GEOMETRY_CONTROLS["logical_metal"]},
        }
    ],
}

# %% [markdown]
# ## Configure Problem And EPR

# %%
sim = EigenmodeSim()
sim.set_geometry(component)
sim.set_stack(stack)
sim.set_output_dir(OUTPUT_CONTROLS["run_dir"])
sim.set_airbox(margin_x=20, margin_y=20, z_above=15)
sim.set_surface_epr(representation=GEOMETRY_CONTROLS["route"], specs=EPR_SPECS)
sim.add_port(
    GEOMETRY_CONTROLS["port_name"],
    layer=GEOMETRY_CONTROLS["logical_metal"],
    layout_sheet=True,
    inductance=GEOMETRY_CONTROLS["port_inductance_h"],
)
sim.set_eigenmode(
    num_modes=SOLVER_CONTROLS["num_modes"],
    target=SOLVER_CONTROLS["target"],
    tolerance=SOLVER_CONTROLS["tolerance"],
    save=SOLVER_CONTROLS["save"],
)
sim.set_numerical(
    **MESH_CONTROLS,
    **{
        key: SOLVER_CONTROLS[key]
        for key in (
            "order",
            "tolerance",
            "max_iterations",
            "solver_type",
            "preconditioner",
            "device",
        )
    },
)

# %% [markdown]
# ## Build Mesh

# %%
run_dir = OUTPUT_CONTROLS["run_dir"]
mesh_path = sim.mesh()
assert mesh_path.name == "palace.msh"
assert f"$MeshFormat\n{VALIDATION_CONTROLS['msh_version']} 0 8" in mesh_path.read_text(
    encoding="utf-8"
)

# %% [markdown]
# ## Write And Validate Config

# %%
config_path = sim.write_config()
config = json.loads(config_path.read_text(encoding="utf-8"))
port_information = json.loads(
    (run_dir / "metadata" / "port_information.json").read_text(encoding="utf-8")
)
index_map = json.loads(
    (run_dir / "metadata" / "palace_index_map.json").read_text(encoding="utf-8")
)
material_resolution = json.loads(
    (run_dir / "metadata" / "palace_material_resolution.json").read_text(
        encoding="utf-8"
    )
)
mesh_manifest = json.loads(
    (run_dir / "metadata" / "mesh_manifest.json").read_text(encoding="utf-8")
)
ms_groups = [
    group for group in mesh_manifest["groups"] if group.get("interface_type") == "MS"
]
assert not ms_groups
port_entry = next(
    entry
    for entry in index_map["entries"]
    if entry["section"] == "Boundaries.LumpedPort"
)
owners = port_entry["metadata"]["owner_semantic_ids"]
embedded_volume_id = port_entry["metadata"]["embedded_volume_id"]
assert len(owners) == len(set(owners)) == VALIDATION_CONTROLS["port_owner_count"]
assert embedded_volume_id
embedded_volumes = [
    volume
    for volume in material_resolution["solution_volumes"]
    if volume["solution_volume"] == embedded_volume_id
]
assert len(embedded_volumes) == VALIDATION_CONTROLS["embedded_solution_volume_count"]
assert embedded_volumes[0]["material"]["kind"] == "vacuum"
lumped_port = config["Boundaries"]["LumpedPort"]
assert len(lumped_port) == len(port_information["ports"]) == 1
assert lumped_port[0]["Direction"] == GEOMETRY_CONTROLS["port_direction"]
assert lumped_port[0]["L"] == GEOMETRY_CONTROLS["port_inductance_h"]

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
assert handoff["status"] == "prepared" and handoff["problem"] == "Eigenmode"
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
    handoff["palace_identity"]["runtime"].removeprefix("v")
    == PROVENANCE["palace_runtime"]
)
assert (
    handoff["palace_identity"]["config_schema"].removeprefix("v")
    == PROVENANCE["palace_schema"]
)

# %% [markdown]
# ## Physics Analysis Results
# Not run: no Palace solve, eigenfrequencies, fields, Q, or participation results exist.

# %% [markdown]
# ## Simulation Performance / Benchmarks

# %%
preparation_facts = {
    "mesh_bytes": mesh_path.stat().st_size,
    "mesh_groups": len(
        json.loads(
            (run_dir / "metadata" / "mesh_manifest.json").read_text(encoding="utf-8")
        )["groups"]
    ),
    "handoff_archive_bytes": plan.archive_path.stat().st_size,
    "data_classification": PROVENANCE["data_classification"],
    "runtime_authority": handoff["source_revisions"]["scgsim_sgb"],
    "solver_cost": "unavailable: Palace was not run",
}
preparation_facts  # noqa: B018
