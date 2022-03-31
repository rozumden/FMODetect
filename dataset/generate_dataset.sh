# TODO: change it to your own dataset directory
BASE_PATH=/cluster/scratch/denysr/dataset
# BASE_PATH=/mnt/lascar/rozumden/dataset

PATTERNS_PATH="${BASE_PATH}/patterns"

BG_PATH="${BASE_PATH}/vot"
DATASET_PATH="${BASE_PATH}/votfmomedtraj_inputs.hdf5"

mkdir -p  ${PATTERNS_PATH} 
mkdir -p  ${BG_PATH} 

VOT_YEAR=2016 bash trackdat/scripts/download_vot.sh dl/vot
bash trackdat/scripts/unpack_vot.sh dl/vot ${BG_PATH}
python3 generate_patterns.py --fg_path ${PATTERNS_PATH} 
python3 process_dataset_traj.py --bg_path ${BG_PATH} --fg_path ${PATTERNS_PATH} --dataset_path ${DATASET_PATH} --dataset_size 5000


