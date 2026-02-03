#!/bin/bash

#SBATCH --partition=day
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --mem=340G
#SBATCH --time=24:00:00
#SBATCH -o L50_batch_mpi.%A.out
#SBATCH --mail-type=ALL --mail-user=ethlau@gmail.com

#module reset
module load miniconda
conda init bash
conda activate py3

export OMP_NUM_THREADS=1

# Define variables
sset="1P"
nparams=35
runs=("n2" "n1" "0" "1" "2")
special_runs_15=("0" "1" "2" "3" "4")
special_runs_29=("n8" "n6" "n4" "n2" "n1" "0" "1" "2")
special_runs_30_32=("n8" "n6" "n4" "n2" "n1" "0" "1" "2")
start_run=1
end_run=35

MPI_flag='--mca orte_base_help_aggregate 0' 

# Loop over the sets
for ((set="$start_run"; set<="$end_run"; set++)); do
    if [ "$set" -eq 15 ]; then
        for run in "${special_runs_15[@]}"; do
            echo "Processing: sset=${sset}_p${set}, run=$run"

            mpirun -np $SLURM_NTASKS python generate_profiles_cy_mpi.py ${sset}_p${set}_${run}
        done

    elif [ "$set" -eq 29 ]; then
        for run in "${special_runs_29[@]}"; do
            echo "Processing: sset=${sset}_p${set}, run=$run"

            mpirun -np $SLURM_NTASKS python generate_profiles_cy_mpi.py ${sset}_p${set}_${run}
        done

    elif [ "$set" -eq 30 ]; then
        for run in "${special_runs_30_32[@]}"; do
            echo "Processing: sset=${sset}_p${set}, run=$run"

            mpirun -np $SLURM_NTASKS python generate_profiles_cy_mpi.py ${sset}_p${set}_${run}
        done

    elif [ "$set" -eq 32 ]; then
        for run in "${special_runs_30_32[@]}"; do
            echo "Processing: sset=${sset}_p${set}, run=$run"

            mpirun -np $SLURM_NTASKS python generate_profiles_cy_mpi.py ${sset}_p${set}_${run}
        done


    else
        for run in "${runs[@]}"; do
            echo "Processing: sset=${sset}_p${set}, run=$run"

            mpirun -np $SLURM_NTASKS python generate_profiles_cy_mpi.py ${sset}_p${set}_${run}
        done
    fi
done   
