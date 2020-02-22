python -m torch.distributed.launch --nproc_per_node=1 ./scripts/test_instance_seg.py --meta ./default/metadata.bin --log_dir ./default/ ./default/config.ini ./default/seamseg_r50_vistas.tar /media/tarek/c0f263ed-e006-443e-8f2a-5860fecd27b5/frustum-pointnets/azure/image/ /media/tarek/c0f263ed-e006-443e-8f2a-5860fecd27b5/frustum-pointnets/azure/bbox_2d/ --person 3

