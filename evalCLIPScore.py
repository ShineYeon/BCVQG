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
from models.blip_VAE import blip_vqg
from models.blip_multimodalVAE import blip_vqg2
from models.blip import init_tokenizer
from metrics import main as nlp_metrics

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist

import wandb


def evaluate_and_save_json(model, data_loader, args, ref_json_path, cand_json_path):
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

    with torch.no_grad():
        for i, (image_ids, images, questions, answers, categories, qlengths, alengths) in enumerate(data_loader):
            images = images.to(device)

            #print(type(images))

            gen_questions = model(1, images, questions, answers, categories, train=False)
            #gen_questions = model(1, images, questions, answers, categories, qlengths, alengths, train=False)

            for img_id, ref_question, gen_question, category in zip(image_ids, questions, gen_questions, categories):
                key = img_id + "-" + category
                if img_id not in references:
                    references[key] = [ref_question]
                else:
                    references[key].append(ref_question)
                candidates[key] = gen_question

            if i % args.log_step == 0:
                print(f'Step [{i}/{total_steps}] processed.')

    # Save references and candidates to JSON files
    with open(ref_json_path, 'w') as ref_file:
        json.dump(references, ref_file)

    with open(cand_json_path, 'w') as cand_file:
        json.dump(candidates, cand_file)

    print(f"Saved references to {ref_json_path}")
    print(f"Saved candidates to {cand_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqg.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local_rank", type=int, default=0)
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


    parser.add_argument('--add-caption-version', action='store_true', help="캡션 버전 사용 여부")
    parser.add_argument('--add-img-version', action='store_true', help="이미지 버전 사용 여부")
    parser.add_argument('--collate-at-raw', action='store_true', help="캡션을 raw에서 collate?")
    parser.add_argument('--multimodal-version', action='store_true', help="멀티모달 버전 사용 여부")

    # Output json paths
    parser.add_argument('--ref-json-path', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/results/reference.json', help='Path to save reference JSON file')
    parser.add_argument('--cand-json-path', type=str, default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/results/candidate.json', help='Path to save candidate JSON file')

    parser.add_argument('--log-to-wandb', action='store_true', help="캡션 버전 사용 여부")
    parser.add_argument('--wandb-name', type=str, default='Eval-BCVQG', help='WanDB Name')

    args = parser.parse_args()

    yaml = YAML(typ='rt')

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # start a new wandb run to track this script
    if args.local_rank == 0 and args.log_to_wandb:
        run = wandb.init(
            project="Heeyeon-BLIP-VAE-New-Eval",
            name=args.wandb_name,
            config={
                "architecture": "BLIP",
                **vars(args)
            }
        )

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = init_tokenizer()

    if args.multimodal_version is True:
        model = blip_vqg2(tokenizer=tokenizer,
                          pretrained=args.pretrained,
                          image_size=config['image_size'],
                          vit=config['vit'],
                          vit_grad_ckpt=config['vit_grad_ckpt'],
                          vit_ckpt_layer=config['vit_ckpt_layer'],
                          add_type_to_posterior=False)
    else:
        model = blip_vqg(tokenizer=tokenizer,
                         pretrained=args.pretrained,
                         image_size=config['image_size'],
                         vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'],
                         vit_ckpt_layer=config['vit_ckpt_layer'],
                         add_img_version=args.add_img_version,
                         add_caption_version=args.add_caption_version)


    val_dataset = create_dataset(args.val_dataset, config, tokenizer=tokenizer, max_examples=args.max_examples,
                                 istrain=False)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        val_sampler = create_sampler(val_dataset, False, num_tasks, global_rank)
    else:
        val_sampler = None

    val_loader = create_loader(val_dataset, val_sampler,
                               batch_size=config['batch_size_test'],
                               num_workers=args.num_workers,
                               is_train=False)

    evaluate_and_save_json(model, val_loader, args, args.ref_json_path, args.cand_json_path)
