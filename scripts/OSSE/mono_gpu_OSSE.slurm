#!/bin/bash
#SBATCH -A yrf@gpu                  # nom du compte
#SBATCH --job-name=gpu_mono         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=40          # nombre de coeurs à réserver (un quart du noeud)
#SBATCH -C v100-32g                 # partition
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=02:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=gpu_mono%j.out     # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.err      # nom du fichier erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et hérités par défaut
module purge
 
# chargement des modules
module load pytorch-gpu/py3/1.4.0
module load geos/3.7.3
export PYTHONPATH=${HOME}/4DVARNN-DinAE:${PYTHONPATH}
 
# echo des commandes lancées
set -x

# parameters
# 1: opt (nadir/nadirswot)
# 2: lag (0/5)
# 3: type_obs (obs/mod)
# 4: domain (GULFSTREAM/OSMOSIS)
# 5: wMis (0/1/2)
# 6: wCov (True/False)
# 7: NN-type (1:ConvaE/2:GENN)
# 8: solver type (FP/GB)

# copie vers le WORK pour execution
JobID=$RANDOM
rsync -avrz --exclude=figs ${HOME}/4DVARNN-DinAE/ ${WORK}/4DVARNN-DinAE_${JobID}
# modify the yaml file according to parameters
sed -i -e 's|_OPT_|'"${1}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_LAG_|'"${2}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_TYPE_OBS_|'"${3}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_DOMAIN_|'"${4}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_WMIS_|'"${5}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_WCOV_|'"${6}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_PHI_|'"${7}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_SOLVER_|'"${8}"'|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/config_zay.yml
sed -i -e 's|_PATH_|'"${WORK}"'/4DVARNN-DinAE_'"${JobID}"'/scripts/OSSE|g' ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/launch_OSSE_zay.py

# exécution du code
python -u ${WORK}/4DVARNN-DinAE_${JobID}/scripts/OSSE/launch_OSSE_zay.py

# remove copy
rm -rf ${WORK}/4DVARNN-DinAE_${JobID}
