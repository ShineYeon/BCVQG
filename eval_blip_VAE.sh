python -m torch.distributed.launch --nproc_per_node=2 eval_blip_VAE.py \
--world_size 2 \
--distributed True \
--num-workers 8 \
--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset.hdf5 \
--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset_weights.json \
--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
--log-to-wandb True \
--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/BCVQG/best_model.pth \
--seed 1234

#python train.py \
#--distributed False \
#--num-workers 4 \
#--dataset /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_train_dataset.hdf5 \
#--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_val_dataset.hdf5 \
#--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_train_dataset_weights.json \
#--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_val_dataset_weights.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/cat2name.json