#!/bin/bash
#SBATCH -A yrf@gpu                  # nom du compte
#SBATCH --job-name=gpu_mono         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --cpus-per-task=40          # nombre de coeurs à réserver (un quart du noeud)
#SBATCH --partition=gpu_p1          # change de partition (preprost: test; gpu_p1: runs)
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=08:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=gpu_mono%j.out     # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out      # nom du fichier erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et hérités par défaut
module purge
 
# chargement des modules
module load pytorch-gpu/py3/1.4.0
module load geos/3.7.3
export PYTHONPATH=${HOME}/4DVARNN-DinAE:${PYTHONPATH}
 
# echo des commandes lancées
set -x
 
# exécution du code
python launch_OSE.py ${1} ${2}
