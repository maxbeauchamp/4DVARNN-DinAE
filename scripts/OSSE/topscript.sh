#!/bin/sh

tobs=${1}
domain="GULFSTREAM"
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 0 False FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 1 False FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 2 False FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 0 True FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 1 True FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 ${tobs} ${domain} 2 True FP
