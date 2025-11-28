# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun -np 20 /opt/venv/bin/python3 02_thermalise/run.py --config 00_potentials/malerba.fs --Tmin 0 --Tmax 1200 --linspace 25
# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun -np 2 /opt/venv/bin/python3 02_thermalise/run.py --config 00_potentials/malerba.fs --Tmin 0 --Tmax 1200 --linspace 25
# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 02_thermalise/run.py --config 00_potentials/malerba.fs --Tmin 0 --Tmax 1200 --linspace 25

# =============================================================
# LAMMPS Lattice Parameter Exploration (Modified)
# =============================================================
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lammps import lammps
from mpi4py import MPI
from matscipy.calculators.eam import EAM
from matscipy.dislocation import get_elastic_constants

# =============================================================
# PATH SETTINGS
# =============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "000_data"))
                                        
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# =============================================================
# MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# LAMMPS Lattice Parameter Exploration (Headless)
# =============================================================
def lammpsSim(temperature, potential_file, max_steps=100000, check_interval=100, tol=1e-4, sim_comm=None):
    """Run a LAMMPS simulation until equilibrium based on lattice relaxation."""

    if temperature == 0:
        return None

    if sim_comm == None:
        raise ValueError("Comm must not be None")

    lmp = lammps(comm=sim_comm)
    lmp.cmd.log(os.path.join(LOG_DIR, f"log_{temperature}.lammps"))

    # ---------------------------
    # Main Settings
    # ---------------------------
    lmp.cmd.units("metal")
    lmp.cmd.dimension(3)
    lmp.cmd.boundary("p", "p", "p")
    lmp.cmd.atom_style("atomic")
    dt = 0.001
    lmp.cmd.timestep(dt)

    # ---------------------------
    # Load EAM potential
    # ---------------------------
    eam_calc = EAM(potential_file)
    alat, _, _, _ = get_elastic_constants(calculator=eam_calc, symbol="Fe", verbose=False)
    lmp.cmd.pair_coeff("* *", potential_file, "Fe")

    # ---------------------------
    # Lattice
    # ---------------------------
    lmp.cmd.lattice("bcc", alat)
    lmp.cmd.region("box", "block", 0, 10, 0, 10, 0, 10)
    lmp.cmd.create_box(1, "box")
    lmp.cmd.create_atoms(1, "box")

    lmp.cmd.pair_style("eam/fs")
    lmp.cmd.neighbor(2.0, "bin")
    lmp.cmd.neigh_modify("delay", 10, "check", "yes")
    lmp.cmd.group("all", "type", "1")

    # ---------------------------
    # Computes, Dumps, Thermo
    # ---------------------------

    lmp.cmd.thermo_style("custom", "step", "temp", "lx", "ly", "lz", 
                         "press", "pxx", "pyy", "pzz", "ke", "pe", "etotal")
    lmp.cmd.thermo(1)

    lmp.cmd.velocity("all", "create", temperature, np.random.randint(1000, 9999), "mom", "yes", "rot", "yes")
    lmp.cmd.fix(1, "all", "npt", "temp", temperature, temperature, 100.0*dt, "iso", 0.0, 0.0, 1000.0*dt)

    # ---------------------------
    # Run
    # ---------------------------
    lmp.cmd.run(max_steps, "pre", "no", "post", "no")

    lmp.close()


# =============================================================
# ARGUMENT PARSER
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate edge dislocation configuration.")
    parser.add_argument("--config", required=True, help="Path to the potential file (e.g., malerba.fs)")
    parser.add_argument("--Tmin", type=float, required=True, help="Minimum temperature (in K)")
    parser.add_argument("--Tmax", type=float, required=True, help="Maximum temperature (in K)")
    parser.add_argument("--linspace", type=int, required=True, help="Number of temperature points between Tmin and Tmax")
    return parser.parse_args()


# =============================================================
# ENTRY POINT
# =============================================================
def main():

    if rank == 0:
        for directory in [OUTPUT_DIR, LOG_DIR]:
            os.makedirs(directory, exist_ok=True)

        print("Output directories created!...\n")

    comm.Barrier()

    args = parse_args()
    temperatures = np.linspace(args.Tmin, args.Tmax, args.linspace)
    potential_file = os.path.abspath(args.config)
    if not os.path.exists(potential_file):
        raise FileNotFoundError(f"Potential file not found: {potential_file}")

    if rank == 0:
        print("\n===== Lattice Parameter Exploration =====")
        print(f"Potential file : {potential_file}")
        print(f"Tmin           : {args.Tmin} K")
        print(f"Tmax           : {args.Tmax} K")
        print(f"Points         : {args.linspace}")
        print(f"MPI ranks      : {size}")
        print("=========================================\n")
    
    # --- Create a sub-communicator per rank (1 core per simulation) ---
    sim_comm = comm.Split(color=rank, key=rank)  # each rank gets its own comm

    # Distribute temperatures across ranks
    for i, temp in enumerate(temperatures):
        if i % size != rank:
            continue  # not this rank's task
        print(f"[Rank {rank}] Running LAMMPS at T = {temp:.2f} K")
        lammpsSim(temp, potential_file, sim_comm=sim_comm)  # pass sub-comm

    comm.Barrier()

    if rank == 0:
        print("\nAll simulations finished.\n")

if __name__ == "__main__":
    main()
