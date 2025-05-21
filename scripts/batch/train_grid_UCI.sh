#!/bin/bash
# Job name: 
#SBATCH --job-name=DNNHuge
#SBATCH --array=0-99%8  # update after seeing TOTAL_JOBS
#SBATCH --gpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --qos=normal

#SBATCH --output=logsUCIDraft_HugeDNN/%A_%a.out
#SBATCH --error=logsUCIDraft_HugeDNN/%A_%a.err

OPTIMIZER="SGD"
OUTPUT_DIR=
DATA_DIR=

# List datasets
#DATASETS=($(ls -d ${DATA_DIR}/*/ | tail -n +0 | head -n 100))

#Datasets with 0-1000 samples -> 77 datasets
#DATASETS=("breast-cancer-wisc-diag" "pittsburg-bridges-TYPE" "breast-cancer-wisc-prog" "statlog-australian-credit" "pittsburg-bridges-REL-L" "annealing" "heart-va" "glass" "trains" "ilpd-indian-liver" "spect" "cylinder-bands" "hayes-roth" "monks-3" "acute-inflammation" "pima" "lymphography" "wine" "oocytes_trisopterus_states_5b" "ionosphere" "pittsburg-bridges-SPAN" "monks-1" "ecoli" "haberman-survival" "vertebral-column-3clases" "parkinsons" "post-operative" "musk-1" "oocytes_trisopterus_nucleus_2f" "energy-y2" "mammographic" "low-res-spect" "dermatology" "heart-hungarian" "primary-tumor" "vertebral-column-2clases" "synthetic-control" "breast-tissue" "breast-cancer" "fertility" "credit-approval" "audiology-std" "balance-scale" "flags" "breast-cancer-wisc" "monks-2" "libras" "energy-y1" "acute-nephritis" "hepatitis" "tic-tac-toe" "heart-cleveland" "echocardiogram" "conn-bench-vowel-deterding" "spectf" "statlog-german-credit" "zoo" "congressional-voting" "statlog-vehicle" "teaching" "seeds" "lenses" "horse-colic" "iris" "arrhythmia" "molec-biol-promoter" "led-display" "balloons" "lung-cancer" "conn-bench-sonar-mines-rocks" "soybean" "blood" "heart-switzerland" "pittsburg-bridges-MATERIAL" "statlog-heart" "pittsburg-bridges-T-OR-D" "planning" )

#Datasets with 1000-5000 samples -> 26 datasets
#DATASETS=("car" "steel-plates" "oocytes_merluccius_states_2f" "bank" "cardiotocography-10clases" "chess-krvkp" "spambase" "semeion" "contrac" "oocytes_merluccius_nucleus_4d" "wine-quality-white" "hill-valley" "waveform" "statlog-image" "plant-margin" "abalone" "ozone" "wine-quality-red" "molec-biol-splice" "titanic" "waveform-noise" "plant-texture" "cardiotocography-3clases" "plant-shape" "image-segmentation" "yeast" )

#Datasets with 5000-100000 samples -> 17
#DATASETS=("wall-following" "letter" "magic" "optical" "twonorm" "chess-krvk" "musk-2" "statlog-landsat" "connect-4" "mushroom" "pendigits" "thyroid" "nursery" "statlog-shuttle" "ringnorm" "adult" "page-blocks" )

#Dataset with 100000+ samples  -> 1 dataset
DATASETS=("miniboone")
#Experimental datasets sample -> 25 datasets
#DATASETS=( "arrhythmia" "breast-tissue" "flags" "glass" "heart-va" "lenses" "trains" "blood" "conn-bench-sonar-mines-rocks" "echocardiogram" "ionosphere" "molec-biol-promoter" "musk-1" "pittsburg-bridges-SPAN" "primary-tumor" "steel-plates" "hepatitis" "congressional-voting" "contrac" "energy-y2" "heart-switzerland" "ilpd-indian-liver" "lung-cancer" "mammographic" "planning")


NUM_DATASETS=${#DATASETS[@]}

# Define learning rates
NUM_LR=4

# Define depths and widths
DEPTHS=(1 2 3 4 5)
WIDTHS=(32 64 128 256 512)
Z_KARE=(0.1)

NUM_DEPTHS=${#DEPTHS[@]}
NUM_WIDTHS=${#WIDTHS[@]}
NUM_Z_KARE=${#Z_KARE[@]}

# == Define models ==
MODELS=( "MSE" )
N_MODELS=${#MODELS[@]}

# Total number of jobs
TOTAL_JOBS=$(( NUM_DATASETS * NUM_LR * NUM_DEPTHS * NUM_WIDTHS * N_MODELS * NUM_Z_KARE ))

echo "Total jobs: ${TOTAL_JOBS}"

# Check that array index is within range
if [ ${SLURM_ARRAY_TASK_ID} -ge ${TOTAL_JOBS} ]; then
  echo "Error: SLURM_ARRAY_TASK_ID (${SLURM_ARRAY_TASK_ID}) exceeds total combinations (${TOTAL_JOBS})."
  exit 1
fi

# Decode indices
IDX=${SLURM_ARRAY_TASK_ID}

WIDTH_IDX=$(( IDX % NUM_WIDTHS ))
IDX=$(( IDX / NUM_WIDTHS ))

DEPTH_IDX=$(( IDX % NUM_DEPTHS ))
IDX=$(( IDX / NUM_DEPTHS ))

Z_KARE_IDX=$(( IDX % NUM_Z_KARE ))
IDX=$(( IDX / NUM_Z_KARE ))

LR_IDX=$(( IDX % NUM_LR ))
IDX=$(( IDX / NUM_LR ))

MODEL_INDEX=$(( IDX % N_MODELS ))
IDX=$(( IDX / N_MODELS ))

DATASET_IDX=$(( IDX % NUM_DATASETS ))


# Now fetch values
DATASET_PATH="${DATASETS[${DATASET_IDX}]}"
DATASET_NAME=$(basename "${DATASET_PATH}")
DEPTH=${DEPTHS[$DEPTH_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}
Z_KARE=${Z_KARE[$Z_KARE_IDX]}
MODEL=${MODELS[$MODEL_INDEX]}

echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET_NAME}"
echo "Learning rate index: ${LR_IDX}"
echo "Depth: ${DEPTH}"
echo "Width: ${WIDTH}"
echo "Z_KARE: ${Z_KARE}"
echo "Model: ${MODEL}"

# Setup environment
source      #source your conda environment script
conda init
conda activate BDGD
export PYTHONPATH= #Export your python path

#Move to the ntk-kare directors
cd path/to/ntk-kare

# Run Python script
python -u scripts/training/train_UCI.py \
    -dir "${DATA_DIR}"\
    -dataset "${DATASET_NAME}"\
    -lr_index "${LR_IDX}"\
    -opt "${OPTIMIZER}"\
    -depth "${DEPTH}"\
    -width "${WIDTH}"\
    -output "${OUTPUT_DIR}"\
    -model "${MODEL}"\
    -z_kare "${Z_KARE}"\
    -max_tot 130000\
    -min_tot 5000\
