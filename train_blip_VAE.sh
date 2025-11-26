# PC 미적용
python -m torch.distributed.launch --nproc_per_node=2 train_blip_VAE.py \
--world_size 2 \
--distributed True \
--num-workers 8 \
--dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset.hdf5 \
--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset.hdf5 \
--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset_weights.json \
--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset_weights.json \
--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
--save-per-epoch 5 \
--beta_0 1000 \
--beta_warmup 0.001 \
--output_dir model-output/BCVQG \
--log-to-wandb \
--wandb-name BCVQG-original \
--config configs/vqg.yaml

#--caption /mnt/disk2/workspace/heeyeon/datasets/COCO/annotations/captions_train2017.json \
#--add-caption-version \
#--add-img-version
#--multimodal-version
#--caption /mnt/disk2/workspace/heeyeon/datasets/COCO/annotations/captions_train2014.json \
#--add-caption-version False \

# 2단계
#python -m torch.distributed.launch --nproc_per_node=2 train_blip_VAE.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset.hdf5 \
#--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset.hdf5 \
#--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset_weights.json \
#--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset_weights.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
#--save-per-epoch 5 \
#--beta_0 1000 \
#--beta_warmup 0.001 \
#--output_dir model-output/MultiModalVersion-PC2nd \
#--log-to-wandb \
#--wandb-name MultiModalVersion-PC2nd \
#--checkpoint-path model-output/MultiModalVersion-PC1st/best_model.pth \
#--config configs/vqg_2nd.yaml \
#--multimodal-version



## PC적용 - 1단계
#python -m torch.distributed.launch --nproc_per_node=2 train_blip_VAE.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset.hdf5 \
#--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset.hdf5 \
#--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset_weights.json \
#--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset_weights.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
#--log-to-wandb True \
#--beta_0 0.0 \
#--beta_warmup 0.0 \
#--output_dir output/VQG/Img-PC/1023-1st \
#--pretrain-encoder
#
## PC적용 - 2단계
#python -m torch.distributed.launch --nproc_per_node=2 train_blip_VAE.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset.hdf5 \
#--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset.hdf5 \
#--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_train_dataset_weights.json \
#--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/iq_val_dataset_weights.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
#--log-to-wandb True \
#--output_dir output/VQG/Img-PC/1024-2nd \
#--checkpoint-path /mnt/disk2/workspace/heeyeon/BLIP-VQG/output/VQG/Img-PC/1023-1st/best_model.pth


#python train.py \
#--distributed False \
#--num-workers 4 \
#--dataset /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_train_dataset.hdf5 \
#--val-dataset /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_val_dataset.hdf5 \
#--train-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_train_dataset_weights.json \
#--val-dataset-weights /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/iq_val_dataset_weights.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQA_processed/cat2name.json