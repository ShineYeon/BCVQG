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

from data.data_loader import *
from models.blip_vqg import blip_vqg
from models.blip import init_tokenizer
from metrics import main as nlp_metrics
import utils

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist

import wandb

def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        filemode='a',  # append mode
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def evaluate(model, data_loader, args, epoch, istrain=False, n_step=False):
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
    total_loss = 0.0

    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0

    total_cider = 0.0
    total_meteor = 0.0
    total_rouge = 0.0
    total_steps = len(data_loader)
    device = torch.device(args.device)

    if args.eval_steps is not None:
        total_steps = min(len(data_loader), args.eval_steps)

    start_time = time.time()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    with torch.no_grad():
        for i, (images, questions, answers, categories, qlengths, alengths) in enumerate(data_loader):
            # Set mini-batch dataset
            images = images.to(device)

            loss = model(images, questions, answers, categories, qlengths, alengths, train=True)
            gen_questions = model(images, questions, answers, categories, qlengths, alengths, train=False)

            bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge = nlp_metrics(questions, gen_questions)

            total_loss += loss.mean()
            total_bleu1 += bleu1
            total_bleu2 += bleu2
            total_bleu3 += bleu3
            total_bleu4 += bleu4
            total_cider += cider
            total_rouge += rouge
            total_meteor += meteor

            if args.eval_steps is not None and i >= args.eval_steps:
                break

            if i % args.log_step == 0: # eval 과정 print
                delta_time = time.time() - start_time
                start_time = time.time()

                if args.local_rank==0:
                    if istrain == True:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Train-Epoch-loss: %.4f'
                                %(delta_time, epoch, config['max_epoch'], i, total_steps, loss.mean()))
                    elif n_step == False:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Val-Epoch-loss: %.4f'
                              % (delta_time, epoch, config['max_epoch'], i, total_steps, loss.mean()))
                    else:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Val-every-n-step-loss: %.4f'
                              % (delta_time, epoch, config['max_epoch'], i, total_steps, loss.mean()))

    return (total_loss / (i+1), total_bleu1/(i+1), total_bleu2/(i+1), total_bleu3/(i+1), total_bleu4/(i+1), total_cider/(i+1), total_meteor/(i+1), total_rouge/(i+1))


def process_lengths(inputs, pad=0):
    """
    Calculates the length of all the sequences in inputs.
    :param inputs: A batch of tensors containing the question or response
    :param pad:
    :return: A list of their lengths
    """
    lengths = [len(input_str) for input_str in inputs]
    return lengths

def compare_outputs(images, questions, answers, categories, qlengths, alengths, model, args, epoch, num_show=5):
    model.eval()

    with torch.no_grad():
        outputs = model(images, questions, answers, categories, qlengths, alengths, train=False)
        for _ in range(num_show):
            if args.local_rank==0:
                print("          ")
            i = random.randint(0, images.size(0)-1)

            if args.local_rank==0:
                print('Sampled question: %s\n Target question: %s -> %s'
                         % (outputs[i], questions[i], answers[i]))
                print("          ")
                print({"Sampled question": outputs[i],
                        "Target question": questions[i],
                        "Target answer": answers[i]})
                logging.info('Epoch: %d: Sampled question: %s\nTarget question: %s -> %s'
                             %(epoch, outputs[i], questions[i], answers[i]))
                logging.info("          ")


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    tokenizer = init_tokenizer()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Dataset
    if args.local_rank==0:
        print("Building data loader...")


    train_dataset = create_dataset(args.dataset, config, tokenizer=tokenizer, max_examples=args.max_examples, istrain=True)
    val_dataset = create_dataset(args.val_dataset, config, tokenizer=tokenizer, max_examples=args.max_examples, istrain=False)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = create_sampler(train_dataset, True, num_tasks, global_rank)
        val_sampler = create_sampler(val_dataset, False, num_tasks, global_rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = create_loader(train_dataset, train_sampler,
                                 batch_size=config['batch_size_train'],
                                 num_workers=args.num_workers,
                                 is_train=True)
    val_loader = create_loader(val_dataset, val_sampler,
                               batch_size=config['batch_size_test'],
                               num_workers=args.num_workers,
                               is_train=False)

    if args.local_rank==0:
        print("Done")
        print("Creating model")

    model = blip_vqg(tokenizer=tokenizer,
                     pretrained=config['pretrained'],
                     image_size=config['image_size'],
                     vit=config['vit'],
                     vit_grad_ckpt=config['vit_grad_ckpt'],
                     vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    if args.local_rank==0:
        print("Done")


    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best_val_loss = 100.0
    best_epoch = 0

    # Train model
    if args.local_rank==0:
        print("Start training")
    start_time = time.time()
    total_steps = len(train_loader)
    n_steps = 0
    for epoch in range(config['max_epoch']):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        for i, (images, questions, answers, categories, qlengths, alengths) in enumerate(train_loader):
            n_steps += 1

            # Set mini-batch dataset
            images = images.to(device)

            # Eval now - using train dataset
            if (args.eval_every_n_steps is not None and \
                    n_steps >= args.eval_every_n_steps and \
                    n_steps % args.eval_every_n_steps == 0):

                if args.local_rank == 0:
                    print('=' * 80)

                start_time = time.time()
                val_loss, bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge = evaluate(model, val_loader, args, epoch, istrain=False, n_step=True)
                delta_time = time.time() - start_time

                if args.local_rank == 0 and args.log_to_wandb == True:
                    print("Time: %.4f, Epoch[%d/%d], Val-every-n-steps-loss: :%.4f" %
                          (delta_time, epoch, config['max_epoch'], val_loss))
                    wandb.log({"Val-every-n-steps-loss": val_loss,
                               "Val-every-n-steps-BLEU1": bleu1,
                               "Val-every-n-steps-BLEU2": bleu2,
                               "Val-every-n-steps-BLEU3": bleu3,
                               "Val-every-n-steps-BLEU4": bleu4,
                               "Val-every-n-steps-CIDEr": cider,
                               "Val-every-n-steps-METEOR": meteor,
                               "Val-every-n-steps-ROUGE": rouge,
                               })

                #compare_outputs(images, questions, answers, categories, qlengths, alengths, model, args, epoch=epoch)

                model.eval()
                with torch.no_grad():
                    outputs = model(images, questions, answers, categories, qlengths, alengths, train=False)
                    for _ in range(5):
                        if args.local_rank == 0:
                            print("          ")
                        idx = random.randint(0, images.size(0) - 1)

                        if args.local_rank == 0:
                            print('Sampled question: %s\n Target question: %s -> %s'
                                  % (outputs[idx], questions[idx], answers[idx]))
                            print("          ")
                            print({"Sampled question": outputs[idx],
                                   "Target question": questions[idx],
                                   "Target answer": answers[idx]})
                            logger.info('Epoch: %d: Sampled question: %s\nTarget question: %s -> %s'
                                         % (epoch, outputs[idx], questions[idx], answers[idx]))
                            logger.info("          ")


            # Forward
            model.train()
            optimizer.zero_grad()

            loss = model(images, questions, answers, categories, qlengths, alengths, train=True)
            loss.mean().backward()
            optimizer.step()

            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                if args.local_rank==0 and args.log_to_wandb:
                    print('Time: %.4f, Epoch [%d/%d], Step [%d/%d], LR: %f, Overall loss: %.4f'
                        %(delta_time, epoch, config['max_epoch'], i, total_steps, optimizer.param_groups[0]["lr"], loss.mean()))
                    # wandb.log({"Time": delta_time,
                    #            "Epoch": '[%d/%d]'%(epoch, config['max_epoch']),
                    #            "Step": '[%d/%d]'%(i, total_steps),
                    #            "LR": optimizer.param_groups[0]["lr"],
                    #            "loss": loss})
                    wandb.log({"Overall-loss": loss.mean()})

            # Save the models
            if args.save_step is not None and (i+1) % args.save_step == 0 and utils.is_main_process():
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj,
                           os.path.join(args.output_dir, 'checkpoint-%02d-%02d.pth'%(epoch+1, i+1)))

        if utils.is_main_process():
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % (epoch + 1)))

        dist.barrier()

        # 1 epoch 끝난 후 evaluate : Train, Valid 데이터셋에 대해
        if args.local_rank == 0:
            print('=' * 80)
        start_time = time.time()
        val_epoch_loss, val_bleu1, val_bleu2, val_bleu3, val_bleu4, val_cider, val_meteor, val_rouge = evaluate(model, val_loader, args, epoch, istrain=False)
        val_delta_time = time.time() - start_time # valid dataset evaluate 소요 시간
        start_time = time.time()
        train_epoch_loss, train_bleu1, train_bleu2, train_bleu3, train_bleu4, train_cider, train_meteor, train_rouge = evaluate(model, train_loader, args, epoch, istrain=False)
        train_delta_time = time.time() - start_time # train dataset evaluate 소요 시간
        if args.local_rank == 0 and args.log_to_wandb:
            print(f"Time: %.4f, Epoch[%d/%d], Train-Epoch-loss: :%.4f" %
                  (train_delta_time, epoch, config['max_epoch'], train_epoch_loss))
            wandb.log({"Train-Epoch-loss": train_epoch_loss,
                       "Train-Epoch-BLEU": train_bleu1,
                       "Train-Epoch-BLEU": train_bleu2,
                       "Train-Epoch-BLEU": train_bleu3,
                       "Train-Epoch-BLEU": train_bleu4,
                       "Train-Epoch-CIDEr": train_cider,
                       "Train-Epoch-METEOR": train_meteor,
                       "Train-Epoch-ROUGE": train_rouge,
                       })
            print('=' * 80)
            print(f"Time: %.4f, Epoch[%d/%d], Val-Epoch-loss: :%.4f" %
                  (val_delta_time, epoch, config['max_epoch'], val_epoch_loss))
            wandb.log({"Val-Epoch-loss": val_epoch_loss,
                       "Val-Epoch-BLEU": val_bleu1,
                       "Val-Epoch-BLEU": val_bleu2,
                       "Val-Epoch-BLEU": val_bleu3,
                       "Val-Epoch-BLEU": val_bleu4,
                       "Val-Epoch-CIDEr": val_cider,
                       "Val-Epoch-METEOR": val_meteor,
                       "Val-Epoch-ROUGE": val_rouge,
                       })

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            if utils.is_main_process():
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"Best model saved at epoch {epoch} with validation loss {best_val_loss}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqg.yaml')
    parser.add_argument('--output_dir', default='output/VQG/0703')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument("--local_rank", type=int, default=0)

    # Session parameters
    parser.add_argument('--log-step', type=int, default=10,
                        help='Step size for printing log info')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval-every-n-steps', type=int, default=None,
                        help='Run eval after every N steps.')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_train_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_val_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='/home/heeyeon/Documents/datasets/VQA_split/cat2name.json',
                        help='Location of mapping from category to type name.')

    parser.add_argument('--log-to-wandb', type=bool,
                        default=True,
                        help='Log or not to wandb')

    args = parser.parse_args()

    yaml = YAML(typ='rt')

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    output_path = os.path.join(args.output_dir, 'config.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    # start a new wandb run to track this script
    if args.local_rank==0 and args.log_to_wandb:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="Heeyeon-BLIP-VQG",
            name="0901-국내저널Figure제작용",
            # track hyperparameters and run metadata
            config={
                "architecture": "BLIP",
                **config,
                **vars(args)
            }
        )

    train(args, config)
