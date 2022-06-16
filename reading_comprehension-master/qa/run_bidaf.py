# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import json
from dataclasses import dataclass
from typing import Any, List, Sequence
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (BertForMultipleChoice, BertConfig, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .load_data import read_race_examples, read_onestop
from . import bidaf
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)



def load_output_model(model, name):
    model.load_state_dict(torch.load(name))

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir",
    #                     default=None,
    #                     type=str,
    #                     required=True,
    #                     help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--d', type=int, default=10)
    parser.add_argument('--max_features', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--run-yev', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--max_batches', type=int, default=None)
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    features = torch.load(Path(__file__).parent.parent / 'data/RACE/features.pickle')
    if args.max_features is not None:
        features = bidaf.feature_subset(features, args.max_features)

    if args.do_train:
        #train_examples = load_data(args, is_training=True)

        num_train_optimization_steps = int(
            features.context.size(0) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # model = BertForMultipleChoice.from_pretrained(args.bert_model,
    #                                               cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
    #                                                                      'distributed_{}'.format(args.local_rank)),
    #                                               num_choices=4)
    model = bidaf.BidafBert(args.d)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

    if args.finetune:
        load_output_model(model, os.path.join(args.output_dir, 'to_finetune.bin'))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        # Load input
        # train_features = convert_examples_to_features(
        #     train_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", features.context.size(0))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        # all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        # all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        # all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        # all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(features.context, features.context_mask, features.context_seq, features.query, features.query_mask, features.query_seq, features.label)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        n_trained_batches = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                n_trained_batches += 1
                if args.max_batches is not None:
                    if n_trained_batches > args.max_batches:
                        break
                batch = tuple(t.to(device) for t in batch)
                context_input_ids, context_input_mask, context_segment_ids, query_input_ids, query_input_mask, query_segment_ids, label_ids = batch
                loss = model(context_input_ids, context_input_mask, context_segment_ids, query_input_ids, query_input_mask, query_segment_ids, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += context_input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        # with open(output_config_file, 'w') as f:
        #     f.write(model_to_save.config.to_json_string())
        # Load a trained model and config that you have fine-tuned
        # config = BertConfig(output_config_file)
        # model = BertForMultipleChoice(config, num_choices=4)
        # model.load_state_dict(torch.load(output_model_file))
    else:
        model = BertForMultipleChoice.from_pretrained(args.bert_model, num_choices=4)
        load_output_model(model, output_model_file)
        # model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_features = torch.load(Path(__file__).parent.parent / 'data/RACE/features_test.pickle')
        if args.max_features is not None:
            eval_features = bidaf.feature_subset(eval_features, args.max_features)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", eval_features.context.size(0))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_data = TensorDataset(eval_features.context, eval_features.context_mask, eval_features.context_seq, eval_features.query, eval_features.query_mask, eval_features.query_seq, eval_features.label)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_results = []
        for input_ids, input_mask, segment_ids, query_input_ids, query_mask, query_segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            query_input_ids = query_input_ids.to(device)
            query_segment_ids = query_segment_ids.to(device)
            query_mask = query_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, input_mask, segment_ids, query_input_ids, query_mask, query_segment_ids, labels=label_ids)
                logits = model(input_ids, input_mask, segment_ids, query_input_ids, query_mask, query_segment_ids)

            logits = logits.detach().cpu().numpy()
            eval_results.append(logits)
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        output_eval_pickle = os.path.join(args.output_dir, "eval_results.pickle")
        torch.save(eval_results, output_eval_pickle)
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  }
        if args.do_train:
            try:
                result['loss'] = tr_loss/nb_tr_steps
            except UnboundLocalError:
                if args.do_train:
                    print("Training loss not available")
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
