#!/bin/sh

# ConvAE
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 1 True 1 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 1 True 1 FP

# GENN (GULFSTREAM)

# nadir / mod
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod GULFSTREAM 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadir 5 mod GULFSTREAM 1 True 2 FP

# nadir / obs
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs GULFSTREAM 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadir 5 obs GULFSTREAM 1 True 2 FP

# nadirswot / mod
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod GULFSTREAM 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadirswot 5 mod GULFSTREAM 1 True 2 FP

# nadirswot / obs
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs GULFSTREAM 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadirswot 5 obs GULFSTREAM 1 True 2 FP

# GENN (OSMOSIS)

# nadir / mod
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 mod OSMOSIS 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadir 5 mod OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 5 mod OSMOSIS 1 False 2 FP

# nadir / obs
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 0 obs OSMOSIS 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadir 5 obs OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadir 5 obs OSMOSIS 1 False 2 FP

# nadirswot / mod
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 mod OSMOSIS 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadirswot 5 mod OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 5 mod OSMOSIS 1 False 2 FP

# nadirswot / obs
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 0 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 1 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 2 False 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 0 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 0 obs OSMOSIS 2 True 2 FP
# ++ nadlag 5
sbatch mono_gpu_OSSE.slurm nadirswot 5 obs OSMOSIS 1 True 2 FP
sbatch mono_gpu_OSSE.slurm nadirswot 5 obs OSMOSIS 1 False 2 FP

