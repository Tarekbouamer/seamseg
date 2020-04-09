N_GPUS=1
DATA_DIR='/media/tarek/Dataset/Seamseg_data/Cityscape/data/'
python setup.py install
clear
python -m torch.distributed.launch --nproc_per_node=$N_GPUS scripts/train_panoptic.py --log_dir ./logs/ ./logs/config.ini $DATA_DIR

