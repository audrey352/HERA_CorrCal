#!/bin/bash
#SBATCH --job-name=corrcal_test
#SBATCH --output=/lustre/aoc/projects/hera/abernier/logs/corrcal_%j_%a.out
#SBATCH --mail-user=audreanne.bernier@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --partition=hera
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-49%5

source ~/.bashrc
conda activate hera3


DAY=$1
DIR=/lustre/aoc/projects/hera/h6c-analysis/IDR2
FILENAMES=/users/abernier/lustre/personal/project/filenames/files_to_calibrate_${DAY}.txt
FRAC=$(awk -v n="$((${SLURM_ARRAY_TASK_ID} + 1))" 'NR==n {print; exit}' ${FILENAMES})
DATA_FILE=${DIR}/${DAY}/zen.${DAY}.${FRAC}.sum.uvh5
MODEL_COV_DIR=/lustre/aoc/projects/hera/rpascua/corrcal/cov_models/h6c
SAVE_LOCATION=/lustre/aoc/projects/hera/abernier/personal/corrcal_results/${DAY}
START_FREQ=149.9
END_FREQ=162.1
POLS=nn
FLAGS_FILE=${DIR}/${DAY}/${DAY}_aposteriori_flags.yaml
INIT_GAINS_FILE=${DIR}/${DAY}/zen.${DAY}.${FRAC}.sum.omni.calfits
MIN_BL_LEN=20
MAX_BL_LEN=150
MIN_GRP_SIZE=5
MAX_ITER=500
BAD_ANTS="8 20 48 62 64 154 181 221 222 238"


echo $(date)

cd /lustre/aoc/projects/hera/abernier/personal/project
echo "calibration.py ${DATA_FILE} ${MODEL_COV_DIR} ${SAVE_LOCATION} \
      --freqs_to_calibrate ${START_FREQ} ${END_FREQ} \
      --pols_to_calibrate ${POLS} \
      --flag_yaml_file ${FLAGS_FILE} \
      --init_gains_file ${INIT_GAINS_FILE} \
      --min_bl_length ${MIN_BL_LEN} \
      --max_bl_length ${MAX_BL_LEN} \
      --min_group_size ${MIN_GRP_SIZE} \
      --maxiter ${MAX_ITER} \
      --extra_bad_ants ${BAD_ANTS}"
python calibration.py ${DATA_FILE} ${MODEL_COV_DIR} ${SAVE_LOCATION} \
      --freqs_to_calibrate ${START_FREQ} ${END_FREQ} \
      --pols_to_calibrate ${POLS} \
      --flag_yaml_file ${FLAGS_FILE} \
      --init_gains_file ${INIT_GAINS_FILE} \
      --min_bl_length ${MIN_BL_LEN} \
      --max_bl_length ${MAX_BL_LEN} \
      --min_group_size ${MIN_GRP_SIZE} \
      --maxiter ${MAX_ITER} \
      --extra_bad_ants ${BAD_ANTS}

echo $(date)
