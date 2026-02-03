# CAMELS_generate_profiles

This repo contains optimized code for calculating gas profiles in cosmological simulations using Cython for significant performance improvements.

## Cython Files

1. `cython_src/process_halos.pyx` - The optimized Cython implementation of the profile calculation function
2. `cython_src/setup.py` - Script to compile the Cython code

### Compile the Cython Module

```bash
cd cython_src
python setup.py build_ext --inplace
```
### 2. Run the Simulation

```bash
mpirun -np <number_of_processes> python generate_profiles_cy_mpi.py <run_name>
```
Where <run_name> is the name of the CAMELS-TNG (L50) run, e.g, `1P_3_n2` . 
An example bash script that loops over the runs is provided in `run_profile_generation_cy.sh`. 
