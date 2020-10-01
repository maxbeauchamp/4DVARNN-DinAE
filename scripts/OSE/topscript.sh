#!/bin/sh

# training with regularization
sbatch mono_gpu_OSE.slurm 0 OSMOSIS False True
sbatch mono_gpu_OSE.slurm 0 GULFSTREAM False True
# training without regularization
sbatch mono_gpu_OSE.slurm 0 OSMOSIS False False
sbatch mono_gpu_OSE.slurm 0 GULFSTREAM False False

