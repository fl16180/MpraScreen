#!/bin/bash
#
#SBATCH --partition=qsu,zihuai,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name="NAME"
#SBATCH --output=data2.txt
#
#SBATCH --time=23:58:00
#SBATCH --mem=100G

# module load python/3.6.1
# source /oak/stanford/groups/zihuai/fredlu/SeqModel/venv/bin/activate
# ml load py-pytorch/1.4.0_py36
# ml load py-numpy/1.17.2_py36
source activate ml
cd /oak/stanford/groups/zihuai/fredlu/MpraScreen/

# python genome_wide_screen.py --job 0 --chr 1 --start 2000 --end 51000 --width 25000
# python extract_features.py
# python setup_project_data.py -p matched_background -r -rb -e -s all
# python setup_neighbor_data.py -p matched_background -e -t E116 -s all
# python setup_mixed_datasets.py --get_genonet


# done
echo "Done"