# STORAGE_PATH=/mnt/lascar/rozumden/dataset
STORAGE_PATH=/cluster/scratch/denysr/dataset
DATASET_PATH="${STORAGE_PATH}/votfmomedtraj.hdf5"
MODEL_PATH="/cluster/home/denysr/tmp/Tensorflow"
python3 train.py --dataset_path ${DATASET_PATH} --model_path ${MODEL_PATH}