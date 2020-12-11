#!/bin/sh

# training with FP
sbatch mono_gpu_OSE.slurm 0 OSMOSIS False FP
sbatch mono_gpu_OSE.slurm 0 GULFSTREAM False FP

