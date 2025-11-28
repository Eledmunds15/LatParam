# LatParam
This is a small github repository containing code that allows you to find the lattice parameter for a specific interatomic potential.

## How it works...
Finding the lattice parameter is actually really easy. All we need to do is create a box full of atoms, leave it at a certain temperature for a while, and then measure the distance between our atoms.

## Instructions


Without MPI:

```
apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 run.py --config 00_potentials/malerba.fs --species Fe --Tmin 50 --Tmax 1200 --linspace 1
```

With MPI:
```
apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun -np 2 /opt/venv/bin/python3 run.py --config 00_potentials/malerba.fs --species Fe --Tmin 50 --Tmax 1200 --linspace 24
```


