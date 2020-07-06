#!/bin/sh

domain="GULFSTREAM"
for type_obs in 'mod' 'obs' ; do
  sbatch mono_gpu_OSSE.slurm swot 0 ${type_obs} ${domain} 0 False
  echo "NN-Learning with SWOT data ("${type_obs}")... Done"
  #for lag in $(seq 0 5); do 
  for lag in '0' '5' ; do
      sbatch mono_gpu_OSSE.slurm nadir ${lag} ${type_obs} ${domain} 0 False
      echo "NN-Learning with NADIR data ("${type_obs}") and lag "${lag}"... Done"
      sbatch mono_gpu_OSSE.slurm nadirswot ${lag} ${type_obs} ${domain} 0 False
      echo "NN-Learning with NADIR/SWOT data ("${type_obs}") and lag "${lag}"... Done"
  done
done

