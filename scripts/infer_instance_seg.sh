
NUM_GPU=1
META_DIR=./default/metadata.bin
LOG_DIR=./default/
CONFIG_DIR=./default/config.ini
WIGHT_FILE=./default/seamseg_r50_vistas.tar
AZURE_FOLDER=/media/tarek/c0f263ed-e006-443e-8f2a-5860fecd27b5/k4a_data/
THRESHOLD=0.9
CLASS=3

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./scripts/test_instance_seg_azure.py --person $CLASS --meta $META_DIR --threshold $THRESHOLD --log_dir $LOG_DIR $CONFIG_DIR $WIGHT_FILE $AZURE_FOLDER