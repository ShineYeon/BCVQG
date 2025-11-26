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

from data.data_loader_with_caption import *
from data.data_loader import *
from models.blip_VAE import blip_vqg
from models.blip_multimodalVAE import blip_vqg2
from models.blip import init_tokenizer
from metrics import main as nlp_metrics
import utils

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import InterpolationMode
import torch.distributed as dist

import wandb

def kl_annealing(step, s=500, k=0.01):
    return float(1 / (1 + np.exp(-k * (step - s))))


def calculate_kl_weight(current_epoch, batch_size, dataset_size, current_iteration):
    # current_global_step 계산
    steps_per_epoch = dataset_size // batch_size
    current_global_step = current_epoch * steps_per_epoch + current_iteration

    # KL weight 계산
    kl_weight = current_global_step / (steps_per_epoch * 50)

    return kl_weight


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


def evaluate(model, beta, data_loader, args, epoch, istrain=False, n_step=False, using_traindataset=False):
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
    total_lmloss = 0.0
    total_klloss = 0.0

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
        for i, batch in enumerate(data_loader):
            if args.add_caption_version is True and using_traindataset is True:
                images, questions, answers, categories, qlengths, alengths, captions = batch
            else:
                images, questions, answers, categories, qlengths, alengths = batch

            # Set mini-batch dataset
            images = images.to(device)

            loss, lm_loss, kl_loss = model(beta, images, questions, answers, categories, qlengths, alengths, train=True)
            gen_questions = model(beta, images, questions, answers, categories, qlengths, alengths, train=False)

            sampled_question = gen_questions[:5]
            target_question = questions[:5]
            target_answer = answers[:5]

            for j in range(len(sampled_question)):
                print('Sampled question: %s\n Target question: %s -> %s' % (
                    sampled_question[j], target_question[j], target_answer[j]))


            bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge = nlp_metrics(questions, gen_questions)

            total_loss += loss
            total_lmloss += lm_loss
            total_klloss += kl_loss

            total_bleu1 += bleu1
            total_bleu2 += bleu2
            total_bleu3 += bleu3
            total_bleu4 += bleu4
            total_cider += cider
            total_meteor += meteor
            total_rouge += rouge

            if args.eval_steps is not None and i >= args.eval_steps:
                break

            if i % args.log_step == 0: # eval 과정 print
                delta_time = time.time() - start_time
                start_time = time.time()

                if args.local_rank==0:
                    if istrain is True:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Train-Epoch-loss: %.4f, LMloss: %.4f, KLloss: %.4f'
                                %(delta_time, epoch, config['max_epoch'], i, total_steps, loss, lm_loss, kl_loss))
                    elif n_step is False:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Val-Epoch-loss: %.4f, LMloss: %.4f, KLloss: %.4f'
                              % (delta_time, epoch, config['max_epoch'], i, total_steps, loss, lm_loss, kl_loss))
                    else:
                        print('Time: %.4f, Epoch [%d/%d], Eval-Step [%d/%d], Val-every-n-step-loss: %.4f, LMloss: %.4f, KLloss: %.4f'
                              % (delta_time, epoch, config['max_epoch'], i, total_steps, loss, lm_loss, kl_loss))

    return (total_loss / (i+1), total_lmloss/(i+1), total_klloss/(i+1), total_bleu1/(i+1), total_bleu2/(i+1), total_bleu3/(i+1), total_bleu4/(i+1), total_cider/(i+1), total_meteor/(i+1), total_rouge/(i+1))


def process_lengths(inputs, pad=0):
    """
    Calculates the length of all the sequences in inputs.
    :param inputs: A batch of tensors containing the question or response
    :param pad:
    :return: A list of their lengths
    """
    lengths = [len(input_str) for input_str in inputs]
    return lengths



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

    # Dataset
    if args.local_rank==0:
        print("Building data loader...")

    if args.add_caption_version is True:
        train_dataset = create_dataset_with_captions(args.dataset, args.caption, config, tokenizer=tokenizer,
                                                     max_examples=args.max_examples, istrain=True)
    else:
        train_dataset = create_dataset(args.dataset, config, tokenizer=tokenizer,
                                       max_examples=args.max_examples, istrain=True)


    val_dataset = create_dataset(args.val_dataset, config, tokenizer=tokenizer, max_examples=args.max_examples, istrain=False)


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = create_sampler(train_dataset, True, num_tasks, global_rank)
        val_sampler = create_sampler(val_dataset, False, num_tasks, global_rank)
    else:
        train_sampler = None
        val_sampler = None

    if args.add_caption_version is True:
        train_loader = create_loader_with_captions(train_dataset, train_sampler,
                                 batch_size=config['batch_size_train'],
                                 num_workers=args.num_workers,
                                 is_train=True)
    else:
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

    if args.multimodal_version is True:
        if args.checkpoint_path is not None:
            model = blip_vqg2(tokenizer=tokenizer,
                              pretrained=args.checkpoint_path,
                              image_size=config['image_size'],
                              vit=config['vit'],
                              vit_grad_ckpt=config['vit_grad_ckpt'],
                              vit_ckpt_layer=config['vit_ckpt_layer'],
                              add_type_to_posterior=False)

        else:
            model = blip_vqg2(tokenizer=tokenizer,
                              pretrained=config['pretrained'],
                              image_size=config['image_size'],
                              vit=config['vit'],
                              vit_grad_ckpt=config['vit_grad_ckpt'],
                              add_type_to_posterior=False)
    else:
        if args.checkpoint_path is not None:
            model = blip_vqg(tokenizer=tokenizer,
                             pretrained=args.checkpoint_path,
                             image_size=config['image_size'],
                             vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'],
                             vit_ckpt_layer=config['vit_ckpt_layer'],
                             denoise_percentage=args.denoise_percentage,
                             add_img_version=args.add_img_version,
                             add_caption_version=args.add_caption_version)

        else:
            model = blip_vqg(tokenizer=tokenizer,
                             pretrained=config['pretrained'],
                             image_size=config['image_size'],
                             vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'],
                             denoise_percentage=args.denoise_percentage,
                             add_img_version=args.add_img_version,
                             add_caption_version=args.add_caption_version)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    if args.local_rank == 0:
        print("Done")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best_val_loss = 100.0
    best_epoch = 0

    # Posterior Collapse: Phase 1
    if args.pretrain_encoder is True:
        if args.distributed:
            model.module.freeze_decoder()
        else:  # for single gpu usage
            model.freeze_decoder()
    # if args.pretrain_encoder:
    #     model.freeze_decoder()

    # Train model
    if args.local_rank == 0:
        print("Start training")
    start_time = time.time()
    total_steps = len(train_loader)
    n_steps = 0

    for epoch in range(config['max_epoch']):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        for i, batch in enumerate(train_loader):
            if args.add_caption_version is True:
                images, questions, answers, categories, qlengths, alengths, captions = batch
            else:
                images, questions, answers, categories, qlengths, alengths = batch

            n_steps += 1

            # Set mini-batch dataset
            images = images.to(device)

            # Posterior Collapse: Phase 1
            if args.pretrain_encoder is True:
                beta = 0
            else:
                # KL annealing
                beta = calculate_kl_weight(epoch, config['batch_size_train'], len(train_dataset), i)
                # beta = kl_annealing(n_steps, s=args.beta_0, k=args.beta_warmup)

            # Eval now - using train dataset
            if (args.eval_every_n_steps is not None and \
                    n_steps >= args.eval_every_n_steps and \
                    n_steps % args.eval_every_n_steps == 0):

                if args.local_rank == 0:
                    print('=' * 80)

                start_time = time.time()
                val_loss, val_lmloss, val_klloss, bleu1, bleu2, bleu3, bleu4, cider, meteor, rouge = evaluate(model, beta, val_loader, args, epoch, istrain=False, n_step=True)
                delta_time = time.time() - start_time

                if args.local_rank == 0 and args.log_to_wandb is True:
                    print("Time: %.4f, Epoch[%d/%d], Val-every-n-steps-loss: :%.4f, LM loss: %.4f, KL loss: %.4f" %
                                (delta_time, epoch, config['max_epoch'], val_loss, val_lmloss, val_klloss))
                    wandb.log({"Val-every-n-steps-loss": val_loss,
                               "Val-every-n-steps-lmloss": val_lmloss,
                               "Val-every-n-steps-klloss": val_klloss,
                                "Val-every-n-steps-BLEU1": bleu1,
                               "Val-every-n-steps-BLEU2": bleu2,
                               "Val-every-n-steps-BLEU3": bleu3,
                               "Val-every-n-steps-BLEU4": bleu4,
                                "Val-every-n-steps-CIDEr": cider,
                                "Val-every-n-steps-METEOR": meteor,
                                "Val-every-n-steps-ROUGE": rouge,
                                })

                if args.local_rank == 0 and args.log_to_wandb is True:
                    with torch.no_grad():
                        if args.add_caption_version:
                            outputs = model(beta, images, questions, answers, categories, qlengths, alengths, captions, train=False,  collate_at_raw=args.collate_at_raw)
                        else:
                            outputs = model(beta, images, questions, answers, categories, qlengths, alengths, train=False)
                        for _ in range(5):
                            print("          ")
                            i = random.randint(0, images.size(0) - 1)

                            sampled_question = outputs[i]
                            target_question = questions[i]
                            target_answer = answers[i]

                            print('Sampled question: %s\n Target question: %s -> %s' % (
                            sampled_question, target_question, target_answer))
                            print("          ")
                            print({"Sampled question": sampled_question,
                                   "Target question": target_question,
                                   "Target answer": target_answer})

                            logging.info('Epoch: %d: Sampled question: %s\nTarget question: %s -> %s' % (
                            epoch, sampled_question, target_question, target_answer))
                            logging.info("          ")

                            # wandb에 로깅
                            wandb.log({
                                "Sampled question": sampled_question,
                                "Target question": target_question,
                                "Target answer": target_answer,
                                "epoch": epoch
                            })


            # Forward
            model.train()
            optimizer.zero_grad()

            if args.add_caption_version is True:
                print("Caption Version")
                loss, lm_loss, kl_loss = model(beta, images, questions, answers, categories, qlengths, alengths, captions, train=True, collate_at_raw=args.collate_at_raw)
            else:
                loss, lm_loss, kl_loss = model(beta, images, questions, answers, categories, qlengths, alengths, train=True)
            #print(loss)
            loss.backward()
            optimizer.step()

            if i % args.log_step == 0:
                delta_time = time.time() - start_time
                start_time = time.time()
                if args.local_rank==0 and args.log_to_wandb:
                    print('Time: %.4f, Epoch [%d/%d], Step [%d/%d], LR: %f, Overall loss: %.4f, LM loss: %.4f, KL loss: %.4f'
                        %(delta_time, epoch, config['max_epoch'], i, total_steps, optimizer.param_groups[0]["lr"], loss , lm_loss, kl_loss))
                    # wandb.log({"Time": delta_time,
                    #            "Epoch": '[%d/%d]'%(epoch, config['max_epoch']),
                    #            "Step": '[%d/%d]'%(i, total_steps),
                    #            "LR": optimizer.param_groups[0]["lr"],
                    #            "loss": loss})
                    wandb.log({"Overall-loss": loss,
                               "LM-loss": lm_loss,
                               "KL-loss": kl_loss})

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

        if utils.is_main_process() and ((epoch+1) % args.save_per_epoch) == 0:
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
        val_epoch_loss, val_epoch_lmloss, val_epoch_klloss, val_bleu1, val_bleu2, val_bleu3, val_bleu4, val_cider, val_meteor, val_rouge = evaluate(model, beta, val_loader, args, epoch, istrain=False)
        val_delta_time = time.time() - start_time # valid dataset evaluate 소요 시간
        start_time = time.time()
        train_epoch_loss, train_epoch_lmloss, train_epoch_klloss, train_bleu1, train_bleu2, train_bleu3, train_bleu4, train_cider, train_meteor, train_rouge = evaluate(model, beta, train_loader, args, epoch, istrain=False, using_traindataset=True)
        train_delta_time = time.time() - start_time # train dataset evaluate 소요 시간
        if args.local_rank == 0 and args.log_to_wandb is True:
            print(f"Time: %.4f, Epoch[%d/%d], Train-Epoch-loss: :%.4f, LM loss: %.4f, KL loss: %.4f" %
                  (train_delta_time, epoch, config['max_epoch'], train_epoch_loss, train_epoch_lmloss, train_epoch_klloss))
            wandb.log({"Train-Epoch-loss": train_epoch_loss,
                       "Train-Epoch-LMloss:": train_epoch_lmloss,
                       "Train-Epoch-KLloss": train_epoch_klloss,
                       "Train-Epoch-BLEU": train_bleu1,
                       "Train-Epoch-BLEU": train_bleu2,
                       "Train-Epoch-BLEU": train_bleu3,
                       "Train-Epoch-BLEU": train_bleu4,
                       "Train-Epoch-CIDEr": train_cider,
                       "Train-Epoch-METEOR": train_meteor,
                       "Train-Epoch-ROUGE": train_rouge,
                       })
            print('=' * 80)
            print(f"Time: %.4f, Epoch[%d/%d], Val-Epoch-loss: :%.4f, LM loss: %.4f, KL loss: %.4f" %
                         (val_delta_time, epoch, config['max_epoch'], val_epoch_loss, val_epoch_lmloss, val_epoch_klloss))
            wandb.log({"Val-Epoch-loss": val_epoch_loss,
                       "Val-Epoch-LMloss": val_epoch_lmloss,
                       "Val-Epoch-KLloss": val_epoch_klloss,
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
    parser.add_argument('--output_dir', default='output/VQG')
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
    parser.add_argument('--save-per-epoch', type=int, default=None,
                        help='Epoch size for saving trained models')

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

    # Caption path
    parser.add_argument('--caption', type=str,
                        default='/mnt/disk2/workspace/heeyeon/datasets/COCO/annotations/captions_train2014.json',
                        help='Path for train annotation json file.')


    # Learning parameters.
    parser.add_argument('--beta_0', type=float, default=1000)
    parser.add_argument('--beta_warmup', type=float, default=0.001)
    parser.add_argument('--top_k', type=float, default=50)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--add-caption-version', action='store_true', help="캡션 버전 사용 여부")
    parser.add_argument('--add-img-version', action='store_true', help="이미지 버전 사용 여부")
    parser.add_argument('--collate-at-raw', action='store_true', help="캡션을 raw에서 collate?")
    parser.add_argument('--multimodal-version', action='store_true', help="멀티모달 버전 사용 여부")



    # Posterior Collapse: Phase 2
    parser.add_argument(
        "--checkpoint-path", help="Checkpoint path", default=None
    )

    # Posterior Collapse: Train Encoder Only
    parser.add_argument( "--pretrain-encoder",
        action="store_true",
        help="Run preliminary encoder training",
    )
    parser.add_argument('--denoise-percentage',
                        type=float,
                        help="Denoise Percentage of input ids",
                        default=0.0
                        )

    parser.add_argument('--log-to-wandb', action='store_true', help="캡션 버전 사용 여부")
    parser.add_argument('--wandb-name', type=str, default='BCVQG', help='WanDB Name')


    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f'Warning: Unrecognized arguments: {unknown}')

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
    if args.local_rank==0 and args.log_to_wandb is True:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="Heeyeon-BLIP-VAE-New",
            name=args.wandb_name,
            # track hyperparameters and run metadata
            config={
                "architecture": "BLIP+VAE-New",
                **config,
                **vars(args)
            }
        )
        #log_file = 'compare_outputs_0626.log'
        #setup_logging(log_file)

    train(args, config)
