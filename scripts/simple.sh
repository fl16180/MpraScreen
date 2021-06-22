#!/bin/bash
#
#SBATCH --partition=qsu,zihuai,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="NAME"
#SBATCH --output=genonet.txt
#
#SBATCH --time=10:00:00
#SBATCH --mem=50G

# module load python/3.6.1
# source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
# ml load py-pytorch/1.4.0_py36
# ml load py-numpy/1.17.2_py36
source activate ml
cd /oak/stanford/groups/zihuai/fredlu/MpraScreen/

# python genome_wide_screen.py --job 0 --chr 1 --start 2000 --end 51000 --width 25000
# python extract_features.py
python setup_project_data.py -p gnom_mpra_mixed -g



# done
echo "Done"