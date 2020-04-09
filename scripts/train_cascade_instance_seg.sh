N_GPUS=1
DATA_DIR='/media/tarek/Dataset/Seamseg_data/Cityscape/data/'
python setup.py install
clear
#python -m torch.distributed.launch --nproc_per_node=$N_GPUS scripts/train_cascade_instance_seg.py --log_dir ./logs/ ./logs/config_cascade.ini $DATA_DIR

python -m torch.distributed.launch --nproc_per_node=$N_GPUS scripts/train_cascade_instance_seg.py --resume ./logs/model_last.pth.tar --log_dir ./logs/ ./logs/config_cascade.ini $DATA_DIR
