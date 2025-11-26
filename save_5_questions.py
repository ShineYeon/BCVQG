import argparse
import os
import numpy as np
import time
from pathlib import Path
from ruamel.yaml import YAML
import utils
from typing import Any, List
from copy import deepcopy
from operator import itemgetter
import math
import logging
import random
import json
import pandas as pd
from prettytable import PrettyTable

from data.data_loader_with_imgId import *
from data.data_loader_with_caption_imgId import *
from models.blip_VAE import blip_vqg
from models.blip_multimodalVAE import blip_vqg2
#from models.blip_vqg import blip_vqg_base
from models.blip import init_tokenizer
from metrics import main as nlp_metrics

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist



def evaluate_and_save_json(model, data_loader, args, ref_json_path, cand_json_path, id_path):
    """
    Calculates VQG average loss on data_loader and saves results as JSON files for CLIPScore evaluation.

    :param model:
    :param data_loader:
    :param args:
    :param ref_json_path: Path to save reference JSON file
    :param cand_json_path: Path to save candidate JSON file
    """

    model.eval()
    total_steps = len(data_loader)
    device = torch.device(args.device)
    model = model.to(device)

    references = {}
    candidates = {}

    ques_id = 0

    id_to_key_path = {}

    with torch.no_grad():

        for i, batch in enumerate(val_loader):
            if args.add_caption_version is True :
                image_ids, images, questions, answers, categories, qlengths, alengths, captions = batch
            else:
                image_ids, images, questions, answers, categories, qlengths, alengths = batch

        # for i, (image_ids, images, questions, answers, categories, qlengths, alengths) in enumerate(val_loader):
            images = images.to(device)

            gen_questions = []
            for j in range(1, 6):
                if args.add_caption_version:
                    gen_question = model(1, images, questions, answers, categories, captions=captions, train=False)
                else:
                    gen_question = model(1, images, questions, answers, categories, train=False)
                gen_questions.append(gen_question)

            gen_questions_per_img = list(zip(*gen_questions))
            assert len(gen_questions_per_img) == len(image_ids)

            for img_id, ref_question, gen_question, category in zip(image_ids, questions, gen_questions_per_img,
                                                                    categories):
                key = img_id + "-" + category + "-" + str(ques_id)
                references[ques_id] = [ref_question]
                candidates[ques_id] = list(gen_question)
                # if key not in references:
                #     references[key] = [ref_question]
                # else:
                #     references[key].append(ref_question)
                # if key not in candidates:
                #     candidates[key] = list(gen_question)
                # else:
                #     candidates[key].append(gen_question)
                id_to_key_path[ques_id] = key
                ques_id += 1
                #candidates[key] = list(gen_question)


            if args.distributed and i % args.log_step == 0:
                print(f'Step [{i}/{total_steps}] processed.')
            if not args.distributed:
                print(f'Step [{i}/{total_steps}] processed.')

    # Save references and candidates to JSON files
    with open(ref_json_path, 'w') as ref_file:
        json.dump(references, ref_file)

    with open(cand_json_path, 'w') as cand_file:
        json.dump(candidates, cand_file)

    with open(id_path, 'w') as id_file:
        json.dump(id_to_key_path, id_file)

    if args.distributed:
        print(f"[Rank {dist.get_rank()}] Saved partial references to {ref_json_path}")
        print(f"[Rank {dist.get_rank()}] Saved partial candidates to {cand_json_path}")

    else:
        print(f"Saved references to {ref_json_path}")
        print(f"Saved candidates to {cand_json_path}")
        print(f"Saved id-key to {id_path}")


def merge_and_save_final(args, world_size):
    """
    After all ranks have produced partial JSON,
    rank=0 merges them into the final JSONs specified by args.ref_json_path / args.cand_json_path
    """
    # rank0 merges partial_{r}.json
    merged_ref = {}
    merged_cand = {}
    merged_id = {}

    for r in range(world_size):
        part_ref = f"{args.ref_json_path}.partial_{r}"
        part_cand = f"{args.cand_json_path}.partial_{r}"
        part_id = f"{args.id_json_path}.partial_{r}"

        if os.path.exists(part_ref):
            with open(part_ref, 'r') as f:
                partial_ref_data = json.load(f)
            for k, v in partial_ref_data.items():
                if k not in merged_ref:
                    merged_ref[k] = v
                else:
                    merged_ref[k].extend(v)
        else:
            print(f"Warning: partial ref file not found: {part_ref}")

        if os.path.exists(part_cand):
            with open(part_cand, 'r') as f:
                partial_cand_data = json.load(f)
            for k, v in partial_cand_data.items():
                if k not in merged_cand:
                    merged_cand[k] = v
                else:
                    merged_cand[k].extend(v)
        else:
            print(f"Warning: partial cand file not found: {part_cand}")

        if os.path.exists(part_id):
            with open(part_id, 'r') as f:
                partial_id_data = json.load(f)
            for k, v in partial_id_data.items():
                if k not in merged_id:
                    merged_id[k] = v
                else:
                    merged_id[k].extend(v)
        else:
            print(f"Warning: partial id file not found: {part_id}")

    # final save
    with open(args.ref_json_path, 'w') as f:
        json.dump(merged_ref, f)
    with open(args.cand_json_path, 'w') as f:
        json.dump(merged_cand, f)
    with open(args.id_json_path, 'w') as f:
        json.dump(merged_id, f)

    print(f"[Rank 0] Merged total {len(merged_ref)} references, {len(merged_cand)} candidates and {len(merged_id)} ids.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqg.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    # Session parameters
    parser.add_argument('--log-step', type=int, default=10, help='Step size for printing log info')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-examples', type=int, default=None, help='For debugging. Limit examples in database.')

    # Data parameters.
    parser.add_argument('--val-dataset', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/iq_val_dataset.hdf5', help='Path for train annotation json file.')
    parser.add_argument('--val-dataset-weights', type=str, default='/home/heeyeon/Documents/datasets/VQA_split/iq_val_dataset_weights.json', help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/data/processed-new/cat2name.json', help='Location of mapping from category to type name.')

    parser.add_argument('--pretrained', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/output/VQG/Add-Img/0910/checkpoint_10.pth')

    parser.add_argument('--base', action='store_true', help="베이스버전 여부")
    parser.add_argument('--add-caption-version', action='store_true', help="캡션 버전 사용 여부")
    parser.add_argument('--add-img-version', action='store_true', help="이미지 버전 사용 여부")
    parser.add_argument('--collate-at-raw', action='store_true', help="캡션을 raw에서 collate?")
    parser.add_argument('--multimodal-version', action='store_true', help="멀티모달 버전 사용 여부")

    # Output json paths
    parser.add_argument('--ref-json-path', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/results/reference.json', help='Path to save reference JSON file')
    parser.add_argument('--cand-json-path', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/results/candidate.json', help='Path to save candidate JSON file')
    parser.add_argument('--id-json-path', type=str,
                        default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/results/reference.json',
                        help='Path to save id JSON file')

    parser.add_argument('--caption', type=str, default='/mnt/disk2/workspace/heeyeon/datasets/COCO/annotations/captions_val2017.json')


    args = parser.parse_args()

    # 1) distributed init
    if args.distributed:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.local_rank
        )
        torch.cuda.set_device(args.local_rank)


    rank = 0
    world_size = 1
    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()


    if args.distributed:
        torch.cuda.manual_seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    else:
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    yaml = YAML(typ='rt')

    with open(args.config, 'r') as f:
        config = yaml.load(f)


    tokenizer = init_tokenizer()

    if args.multimodal_version is True:
        model = blip_vqg2(tokenizer=tokenizer,
                          pretrained=args.pretrained,
                          image_size=config['image_size'],
                          vit=config['vit'],
                          vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'],
                          add_type_to_posterior=False)
    # elif args.base is True:
    #     model = blip_vqg_base(tokenizer=tokenizer,
    #                       pretrained=args.pretrained,
    #                       image_size=config['image_size'],
    #                       vit=config['vit'],
    #                       vit_grad_ckpt=config['vit_grad_ckpt'],
    #                       vit_ckpt_layer=config['vit_ckpt_layer'])
    else:
        model = blip_vqg(tokenizer=tokenizer,
                         pretrained=args.pretrained,
                         image_size=config['image_size'],
                         vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'],
                         add_img_version=args.add_img_version,
                         add_caption_version=args.add_caption_version)

    if args.add_caption_version:
        val_dataset = create_dataset_with_captions(args.val_dataset, args.caption, config, tokenizer=tokenizer, max_examples=args.max_examples,
                                 istrain=False)
    else:
        val_dataset = create_dataset(args.val_dataset, config, tokenizer=tokenizer, max_examples=args.max_examples,
                                 istrain=False)

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        val_sampler = create_sampler(val_dataset, False, num_tasks, global_rank)
    else:
        val_sampler = None

    if args.add_caption_version:
        val_loader = create_loader_with_captions(val_dataset, val_sampler,
                                                 batch_size=config['batch_size_test'],
                                                 num_workers=args.num_workers, is_train=False)
    else:
        val_loader = create_loader(val_dataset, val_sampler,
                               batch_size=config['batch_size_test'],
                               num_workers=args.num_workers,
                               is_train=False)

    if args.distributed:
        partial_ref_file = f"{args.ref_json_path}.partial_{rank}"
        partial_cand_file = f"{args.cand_json_path}.partial_{rank}"
        partial_id_file = f"{args.id_json_path}.partial_{rank}"
        evaluate_and_save_json(model, val_loader, args, partial_ref_file, partial_cand_file, partial_id_file)
    else:
        evaluate_and_save_json(model, val_loader, args, args.ref_json_path, args.cand_json_path, args.id_json_path)

    # barrier
    if args.distributed:
        dist.barrier()

    # 6) Rank0 merges
    if rank == 0:
        merged_ref = {}
        merged_cand = {}
        merged_id = {}

        for r in range(world_size):
            refp = f"{args.ref_json_path}.partial_{r}"
            candp = f"{args.cand_json_path}.partial_{r}"
            idp = f"{args.id_json_path}.partial_{r}"

            if os.path.exists(refp):
                with open(refp, 'r') as f:
                    sub_ref = json.load(f)
                for k, v in sub_ref.items():
                    if k not in merged_ref:
                        merged_ref[k] = v
                    else:
                        merged_ref[k].extend(v)

            if os.path.exists(candp):
                with open(candp, 'r') as f:
                    sub_cand = json.load(f)
                for k, v in sub_cand.items():
                    if k not in merged_cand:
                        merged_cand[k] = v
                    else:
                        merged_cand[k].extend(v)

            if os.path.exists(idp):
                with open(idp, 'r') as f:
                    sub_id = json.load(f)
                for k, v in sub_id.items():
                    if k not in merged_id:
                        merged_id[k] = v
                    else:
                        merged_id[k].extend(v)

        # final save
        with open(args.ref_json_path, 'w') as f:
            json.dump(merged_ref, f)
        with open(args.cand_json_path, 'w') as f:
            json.dump(merged_cand, f)
        with open(args.id_json_path, 'w') as f:
            json.dump(merged_id, f)

        print(f"[Rank 0] Merged final references: {len(merged_ref)} entries, candidates: {len(merged_cand)}, ids: {len(merged_id)}")
        print(f"Saved final references to {args.ref_json_path}")
        print(f"Saved final candidates to {args.cand_json_path}")
        print(f"Saved final ids to {args.id_json_path}")

    # final barrier (optional)
    if args.distributed:
        dist.barrier()

    if rank == 0:
        print("All done!")
    # end
    if args.distributed:
        dist.destroy_process_group()
