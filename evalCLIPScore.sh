#python evalCLIPScore.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
#--cat2name /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/cat2name.json \
#--log-to-wandb False \
#--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/AddCaption/collate-at-raw/best_model.pth \
#--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/AddCaption/collate-at-raw/seed-1234/reference.json \
#--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/AddCaption/collate-at-raw/seed-1234/candidate.json \
#--seed 1234 \
#--add-img-version False \
#--add-caption-version True

python -m torch.distributed.launch --nproc_per_node=2 evalCLIPScore.py \
--world_size 2 \
--distributed True \
--num-workers 8 \
--seed 1234 \
--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/BCVQG/best_model.pth \
--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG/seed-1234/reference.json \
--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG/seed-1234/candidate.json \
--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \

#
#python -m torch.distributed.launch --nproc_per_node=2 evalCLIPScore.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/MultimodalVAE/best_model.pth \
#--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
#--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal/seed-1234/reference.json \
#--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal/seed-1234/candidate.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
#--multimodal-version

