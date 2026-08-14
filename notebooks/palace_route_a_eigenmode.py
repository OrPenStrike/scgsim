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
# # Public OrPen Xmon — Route A Eigenmode
# The authored public junction marker supplies the source sheet and direction.
# The target is the explicit public D1 logical metal group; no second port API
# or name/bbox ownership inference is used.

# %% [markdown]
# ## Design And Geometry Controls

# %%
from pathlib import Path

from orpen_sc_pdk.tech import OUTER_VACUUM_THICKNESS_UM

GEOMETRY_CONTROLS = {
    "route": "A",
    "component": "kosen2024_flip_chip_xmon_qubit",
    "port_name": "o_junction_lumped",
    "logical_metal": "D1_BOTTOM_M1",
    "inductance_h": 1e-12,
    "coupon_padding_um": 75.0,
    "air_below_thickness_um": float(OUTER_VACUUM_THICKNESS_UM),
    "air_above_thickness_um": float(OUTER_VACUUM_THICKNESS_UM),
}

# %% [markdown]
# ## Meshing Controls

# %%
MESH_CONTROLS = {"refined_mesh_size": 15.0, "max_mesh_size": 80.0}

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

# %%
EXECUTION_CONTROLS = {
    "machine_profiles": ("ltlab-local", "ltlab-slurm", "f1-slurm"),
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
try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_DIR = (
        Path.cwd() / "notebooks" if (Path.cwd() / "notebooks").is_dir() else Path.cwd()
    )

OUTPUT_CONTROLS = {
    "run_dir": NOTEBOOK_DIR
    / ".artifacts"
    / "kosen2024_flip_chip_xmon_route_a_eigenmode",
    "output_formats": ("gds", "xao", "msh2", "json", "sbatch", "tar.gz"),
}

# %% [markdown]
# ## Validation And Failure Controls

# %%
VALIDATION_CONTROLS = {
    "msh_version": "2.2",
    "port_owner_count": 2,
    "adjacent_solution_volume_count": 2,
    "no_solver_run": True,
}

# %% [markdown]
# ## Data Classification And Provenance

# %%
PROVENANCE = {
    "classification": "public",
    "orpen_sc_pdk_revision": "a16e8a123ce3ebfbda30aba31024506c2dcfd0c8",
    "gsim_meshing_methodology": "8f5dc6c05255d003a9c6d8959537bcf8068379d3",
    "palace_runtime": "0.16.1",
    "palace_schema": "0.16.0",
}

# %%
import json

import gdsfactory as gf
import orpen_sc_pdk
from orpen_sc_pdk import LAYER, LAYER_STACK, get_material_records

from scgsim.palace import EigenmodeSim
from scgsim.sgb import build_kosen2024_flip_chip_xmon_stack

EPR_SPECS = {
    kind: {"thickness": 0.003, "permittivity": 10.0, "loss_tangent": 0.0}
    for kind in ("MA", "MS", "SA")
}

# %% [markdown]
# ## Build Component

# %%
orpen_sc_pdk.activate()
gf.clear_cache()
component = gf.get_component(GEOMETRY_CONTROLS["component"])
stack = build_kosen2024_flip_chip_xmon_stack(
    component=component,
    layer_stack=LAYER_STACK,
    material_records=get_material_records(),
    d0_top_ground_mask_layer=tuple(LAYER.D0_TOP_GROUND_MASK),
    indium_bump_layer=tuple(LAYER.D0_D1_INDIUM_BUMP),
    coupon_padding_um=GEOMETRY_CONTROLS["coupon_padding_um"],
    air_below_thickness_um=GEOMETRY_CONTROLS["air_below_thickness_um"],
    air_above_thickness_um=GEOMETRY_CONTROLS["air_above_thickness_um"],
)
solution_regions = stack["solution_regions"]
assert tuple(solution_regions) == (
    "AIR_BELOW",
    "D0_SUBSTRATE",
    "D0_TO_D1_GAP",
    "D1_SUBSTRATE",
    "AIR_ABOVE",
)
assert (
    len(
        {
            json.dumps(region["geometry"]["domain_bounds_um"], sort_keys=True)
            for region in solution_regions.values()
        }
    )
    == 1
)
assert (
    solution_regions["AIR_BELOW"]["geometry"]["z_max_um"]
    == solution_regions["D0_SUBSTRATE"]["geometry"]["z_min_um"]
)
assert (
    solution_regions["D1_SUBSTRATE"]["geometry"]["z_max_um"]
    == solution_regions["AIR_ABOVE"]["geometry"]["z_min_um"]
)
run_dir = OUTPUT_CONTROLS["run_dir"]
if run_dir.exists() and any(run_dir.iterdir()):
    raise FileExistsError(f"Preserving existing inspectable run folder: {run_dir}")

# %% [markdown]
# ## Configure Problem And EPR

# %%
sim = EigenmodeSim()
sim.set_geometry(component)
sim.set_stack(stack)
sim.set_output_dir(run_dir)
sim.set_surface_epr(representation=GEOMETRY_CONTROLS["route"], specs=EPR_SPECS)
sim.add_port(
    GEOMETRY_CONTROLS["port_name"],
    layer=GEOMETRY_CONTROLS["logical_metal"],
    layout_sheet=True,
    inductance=GEOMETRY_CONTROLS["inductance_h"],
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
mesh_path = sim.mesh()
assert f"$MeshFormat\n{VALIDATION_CONTROLS['msh_version']} 0 8" in mesh_path.read_text()
manifest = json.loads((run_dir / "metadata" / "mesh_manifest.json").read_text())
port_group = next(
    group for group in manifest["groups"] if group["section"] == "port_surfaces"
)
assert len(port_group["owner_semantic_ids"]) == VALIDATION_CONTROLS["port_owner_count"]
assert (
    len(port_group["adjacent_solution_volume_ids"])
    == VALIDATION_CONTROLS["adjacent_solution_volume_count"]
)

# %% [markdown]
# ## Write And Validate Config

# %%
config_path = sim.write_config()
config = json.loads(config_path.read_text())
assert len(config["Boundaries"]["LumpedPort"]) == 1
port_information = json.loads(
    (run_dir / "metadata" / "port_information.json").read_text()
)
assert port_information["ports"][0]["name"] == GEOMETRY_CONTROLS["port_name"]

# %% [markdown]
# ## Prepare And Inspect Handoff

# %%
handoff = sim.prepare_handoff(
    profile=EXECUTION_CONTROLS["selected_profile"],
    executable=EXECUTION_CONTROLS["executable"],
    resources=EXECUTION_CONTROLS["resources"],
    setup_commands=EXECUTION_CONTROLS["setup_commands"],
)
assert (
    handoff.script_path.name == "run_palace.sbatch" and handoff.archive_path.is_file()
)

# %% [markdown]
# ## Physics Analysis Results
# The prepared archive contains no physics result: Palace is not run here.

# %%
run_status = json.loads((run_dir / "metadata" / "palace_run_metadata.json").read_text())
assert run_status["status"] == "not_run"

# %% [markdown]
# ## Simulation Performance / Benchmarks

# %%
resource_record = json.loads(
    (run_dir / "metadata" / "palace_resource_record.json").read_text()
)
assert resource_record["status"] == "not_submitted"
print(
    {
        "mesh": str(mesh_path),
        "config": str(config_path),
        "handoff": str(handoff.archive_path),
    }
)
