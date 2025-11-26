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

from data.data_loader import *
from models.blip_VAE import blip_vqg
from models.blip import init_tokenizer
from metrics import main as nlp_metrics

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist

import wandb


def evaluate(model, data_loader, args):
    """
    Calculates vqg average loss on data_loader

    :param model:
    :param data_loader:
    :param criterion:
    :param l2_criterion:
    :param args:
    :return: A float value of average loss
    """

    model.eval()
    total_steps = len(data_loader)
    device = torch.device(args.device)

    model = model.to(device)

    start_time = time.time()

    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0

    total_cider = 0.0
    total_meteor = 0.0
    total_rouge = 0.0

    with torch.no_grad():
        for i, (images, questions, answers, categories, qlengths, alengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)

            gen_questions = model(1, images, questions, answers, categories, qlengths, alengths, train=False)

            bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge = nlp_metrics(questions, gen_questions)

            total_bleu1 += bleu1
            total_bleu2 += bleu2
            total_bleu3 += bleu3
            total_bleu4 += bleu4
            total_cider += cider
            total_meteor += meteor
            total_rouge += rouge

            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()

                if args.local_rank == 0:
                    print('Time: %.4f, Step [%d/%d], bleu1: %.4f, bleu2: %.4f, bleu3: %.4f, bleu4: %.4f, cider: %.4f, meteor: %.4f, rouge: %.4f'
                          % (delta_time, i, total_steps, bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge))
                    wandb.log({"test-BLEU1": bleu1,
                               "test-BLEU2": bleu2,
                               "test-BLEU3": bleu3,
                               "test-BLEU4": bleu4,
                               "test-CIDEr": cider,
                               "test-METEOR": meteor,
                               "test-ROUGE": rouge})

    results = {
        'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'CIDEr', 'METEOR', 'ROUGE'],
        'Score': [total_bleu1/total_steps, total_bleu2/total_steps, total_bleu3/total_steps, total_bleu4/total_steps, total_cider/total_steps, total_meteor/total_steps, total_rouge/total_steps]
    }

    # DataFrame으로 변환
    df = pd.DataFrame(results)

    # PrettyTable을 사용한 출력
    table = PrettyTable()
    table.field_names = ["Metric", "Score"]

    for index, row in df.iterrows():
        table.add_row([row['Metric'], row['Score']])

    print(table)
    logging.info(table)

    return (total_bleu1 / total_steps, total_bleu2/total_steps, total_bleu3/total_steps, total_bleu4/total_steps, cider / total_steps, meteor / total_steps, rouge / total_steps)


def process_lengths(inputs, pad=0):
    """
    Calculates the length of all the sequences in inputs.
    :param inputs: A batch of tensors containing the question or response
    :param pad:
    :return: A list of their lengths
    """
    lengths = [len(input_str) for input_str in inputs]
    return lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqg.yaml')
    parser.add_argument('--output_dir', default='output/VQG')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)

    # Session parameters
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for printing log info')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Data parameters.
    parser.add_argument('--val-dataset', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_val_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/cat2name.json',
                        help='Location of mapping from category to type name.')

    parser.add_argument('--log-to-wandb', type=bool,
                        default=True,
                        help='Log or not to wandb')

    parser.add_argument('--pretrained', type=str,
                        default='/mnt/disk2/workspace/heeyeon/BLIP-VQG/output/VQG/PC/0802_2nd/best_model.pth')

    parser.add_argument('--add-img-version', type=bool, default=False)

    args = parser.parse_args()

    yaml = YAML(typ='rt')

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # start a new wandb run to track this script
    if args.local_rank==0 and args.log_to_wandb:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="Heeyeon-BLIP-VAE-New",
            name="0803PC-Evaluate",
            # track hyperparameters and run metadata
            config={
                "architecture": "BLIP",
                **vars(args)
            }
        )

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = init_tokenizer()
    model = blip_vqg(tokenizer=tokenizer,
                     pretrained=args.pretrained,
                     image_size=224,
                     vit='base',
                     vit_grad_ckpt=False,
                     vit_ckpt_layer=0,
                     add_img_version=args.add_img_version)


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

    evaluate(model, val_loader, args)