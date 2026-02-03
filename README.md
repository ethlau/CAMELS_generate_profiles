# CAMELS_generate_profiles

This repo contains optimized code for calculating gas profiles in cosmological simulations using Cython for significant performance improvements.

## Requirements
1. cython
2. illustris_python
3. mpi4py
4. astropy, scipy, numpy, h5py, tqdm

## How to use: 

### Compile the Cython Module

```bash
cd cython_src
python setup.py build_ext --inplace
```
### 2. Run the Code

```bash
mpirun -np <number_of_processes> python generate_profiles_cy_mpi.py <run_name>
```
Where <run_name> is the name of the CAMELS-TNG (L50) run, e.g, `1P_3_n2` . 
An example bash script that loops over the runs is provided in `run_profile_generation_cy.sh`. 
