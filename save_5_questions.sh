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

#python -m torch.distributed.launch --nproc_per_node=2 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/BCVQG/best_model.pth \
#--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
#--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG/seed-1234/reference-5Q.json \
#--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG/seed-1234/candidate-5Q.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
#
#python -m torch.distributed.launch --nproc_per_node=2 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/BCVQG-PC2nd/best_model.pth \
#--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
#--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG-PC/seed-1234/reference-5Q.json \
#--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/BCVQG-PC/seed-1234/candidate-5Q.json \
#--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \

#python -m torch.distributed.launch --nproc_per_node=2 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /mnt/disk2/workspace/heeyeon/BLIP-VQG/output/VQG/baseline/best_model.pth \
#--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
#--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions-new/baseline/reference-5Q-test.json \
#--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions-new/baseline/candidate-5Q-test.json \
#--id-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions-new/baseline/id-path-test.json \
#--cat2name /mnt/disk2/workspace/heeyeon/BCVQG/BLIP-VQG/data/processed-new/cat2name.json
#
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/BCVQG-PC/best_model.pth \
#--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
#--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/BCVQG-PC/reference-5Q.json \
#--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/BCVQG-PC/candidate-5Q.json \
#--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/BCVQG-PC/id-path.json \
#--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json
#
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/AddImg/best_model.pth \
#--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
#--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg/reference-5Q.json \
#--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg/candidate-5Q.json \
#--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg/id-path.json \
#--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
#--add-img-version
#
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/AddImg-PC/best_model.pth \
#--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
#--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg-PC/reference-5Q.json \
#--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg-PC/candidate-5Q.json \
#--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddImg-PC/id-path.json \
#--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
#--add-img-version
#
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29501 save_5_questions.py \
--world_size 3 \
--distributed True \
--num-workers 12 \
--seed 1234 \
--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/AddCaption/best_model.pth \
--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption/0530reference-5Q.json \
--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption/0530candidate-5Q.json \
--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption/0530id-path.json \
--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
--add-caption-version \
--caption /home/compu/heeyeon/BCVQG/new-BCVQG/data/captions_val2017.json

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29501 save_5_questions.py \
--world_size 3 \
--distributed True \
--num-workers 12 \
--seed 1234 \
--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/AddCaption-PC/best_model.pth \
--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption-PC/0530reference-5Q.json \
--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption-PC/0530candidate-5Q.json \
--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/AddCaption-PC/0530id-path.json \
--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
--add-caption-version \
--caption /home/compu/heeyeon/BCVQG/new-BCVQG/data/captions_val2017.json
#
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/Multimodal/best_model.pth \
#--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
#--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal/reference-5Q.json \
#--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal/candidate-5Q.json \
#--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal/id-path.json \
#--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
#--multimodal-version
#
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 save_5_questions.py \
#--world_size 2 \
#--distributed True \
#--num-workers 8 \
#--seed 1234 \
#--pretrained /home/compu/heeyeon/BCVQG/new-BCVQG/model-output/Multimodal-PC/best_model.pth \
#--val-dataset /home/compu/heeyeon/BCVQG/new-BCVQG/data/iq_val_dataset.hdf5 \
#--ref-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal-PC/reference-5Q.json \
#--cand-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal-PC/candidate-5Q.json \
#--id-json-path /home/compu/heeyeon/BCVQG/new-BCVQG/gen-questions/Multimodal-PC/id-path.json \
#--cat2name /home/compu/heeyeon/BCVQG/new-BCVQG/data/cat2name.json \
#--multimodal-version
#
##
##python -m torch.distributed.launch --nproc_per_node=2 evalCLIPScore.py \
##--world_size 2 \
##--distributed True \
##--num-workers 8 \
##--seed 1234 \
##--pretrained /mnt/disk2/workspace/heeyeon/new-BCVQG/model-output/MultimodalVAE/best_model.pth \
##--val-dataset /mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset_new.hdf5 \
##--ref-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal/seed-1234/reference.json \
##--cand-json-path /mnt/disk2/workspace/heeyeon/new-BCVQG/gen-questions/Multimodal/seed-1234/candidate.json \
##--cat2name /mnt/disk2/workspace/heeyeon/datasets/VQG_processed/cat2name.json \
##--multimodal-version
#
