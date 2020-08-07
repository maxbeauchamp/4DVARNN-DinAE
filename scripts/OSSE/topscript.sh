#!/bin/sh

tobs=${1}
domain="GULFSTREAM"
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 0 False
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 1 False
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 2 False
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 0 True
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 1 True
sbatch mono_gpu_OSSE.slurm nadir 0 ${tobs} ${domain} 2 True
