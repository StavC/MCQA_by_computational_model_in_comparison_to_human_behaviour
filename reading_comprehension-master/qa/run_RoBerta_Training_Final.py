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
"""Roberta finetuning runner."""

from __future__ import absolute_import


"""
Most of the code was taken from here :  https://github.com/rodgzilla/pytorch-pretrained-BERT/tree/multiple-choice-code/examples

John's edited the code first and then then Stav and Nurit changed it to run on RoBerta and not BERT

this code should first train on Race and then Finetune more on OneStopQA


Params to send for training, fine tune existing mode, do eval and run on OneStopQA
--RoBerta_model=roberta-base --output_dir=/home/cohnstav/ModelWeights --do_train --do_eval --finetune --run-yev --max_seq_length 512 --num_train_epochs 30


"""


import argparse
import logging
import os
import random
from io import open
from transformers import AdamW

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_pretrained_bert.modeling import (BertForMultipleChoice, BertConfig, WEIGHTS_NAME, CONFIG_NAME)
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig
from transformers import RobertaTokenizer, RobertaForMultipleChoice
import transformers

transformers.logging.set_verbosity_error()
from load_data import read_race_examples, read_onestop

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,

            }
            for _, input_ids, input_mask in choices_features
        ]
        self.label = label


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    ### Roberta tokenization
    """Loads a data file into a list of `InputBatch`s."""

    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.

    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)

            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["<s>"] + context_tokens_choice + ["</s>"] + ending_tokens + ["</s>"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids) + [0] * (max_seq_length - len(input_ids))
            # Zero-pad up to the sequence length.
            padding_ids = [1] * (max_seq_length - len(input_ids))  # 1 for roberta
            input_ids += padding_ids

            choices_features.append((tokens, input_ids, input_mask))

        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            for choice_idx, (tokens, input_ids, input_mask) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            if is_training:
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.id,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def load_data2(args, is_training=True):  # use this when running from home pc
    if args.run_yev:
        print("Using yev data")
        return read_onestop(is_training=is_training)
    elif (not (args.run_yev)) and (is_training == True):
        return read_race_examples(Path(__file__).parent.parent / 'data' / 'RACE' / 'train' / 'combined',
                                  is_training=is_training)
    else:
        return read_race_examples(Path(__file__).parent.parent / 'data' / 'RACE' / 'dev' / 'combined',
                                  is_training=is_training)


def load_data(args, is_training=True):  # use this when running through tmux or home
    if args.run_yev:
        print("Using yev data")
        return read_onestop(is_training=is_training)
    elif (not (args.run_yev)) and (is_training == True):
        return read_race_examples(
            Path('/tmp/pycharm_project_972/reading_comprehension-master/data/RACE/train/combined'), # path to race train middle and high
            is_training=is_training)
    else:
        return read_race_examples(Path('/tmp/pycharm_project_972/reading_comprehension-master/data/RACE/dev/combined'),# path to race dev middle and high
                                  is_training=is_training)


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
    parser.add_argument("--RoBerta_model", default='roberta-base', type=str, required=True,
                        help="roberta-base")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
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
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        # default=5e-5,
                        default=0.000008,  # 0.00001 for roberta on race
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20,
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

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = load_data(args, is_training=True)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

    if args.finetune:
        ##### Race tained model used to finetune on onestop
        load_output_model(model, os.path.join(args.output_dir,
                                              '/home/cohnstav/ModelWeights/robertatraininguptoEpoch11/2acc:0.7403314917127072'))  # 0.7946 on epoch 6
        ##### Race tained model used to finetune on onestop

        # un comment above

    model.to(device)
    print(
        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(n_gpu)

    if n_gpu > 1:
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


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion,
                                                num_training_steps=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_label)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        print('above train')
        model.train()
        n_trained_batches = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                n_trained_batches += 1
                if args.max_batches is not None:
                    if n_trained_batches > args.max_batches:
                        break
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids = batch

                loss = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss.loss / args.gradient_accumulation_steps

                tr_loss += loss
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                loss.backward()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            # now do eval
            eval_examples = load_data(args, is_training=False)
            eval_features = convert_examples_to_features(
                eval_examples, tokenizer, args.max_seq_length, True)
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            eval_results = []
            for input_ids, input_mask, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
                    logits = model(input_ids=input_ids, attention_mask=input_mask)


                logits = logits.logits.detach().cpu().numpy()
                eval_results.append(logits)
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)
                #print(label_ids)

                eval_loss += tmp_eval_loss.loss.mean()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            output_eval_pickle = os.path.join(args.output_dir, f"eval_resultsEpoch{_}.pickle")
            torch.save(eval_results, output_eval_pickle)
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      }
            if args.do_train:
                try:
                    result['loss'] = tr_loss / nb_tr_steps
                except UnboundLocalError:
                    if args.do_train:
                        print("Training loss not available")
            output_eval_file = os.path.join(args.output_dir, f"eval_results epoch:{_}.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # name = f'{_}acc:{eval_accuracy}'
            name = f'{_}acc:{eval_accuracy}OneStopQAFold4.pt'

            output_model_file = os.path.join(args.output_dir, name)

            torch.save(model_to_save.state_dict(), output_model_file)

            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            ####

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        output_model_file = os.path.join(args.output_dir, 'lastepoch')

        ######

        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        # Load a trained model and config that you have fine-tuned
        config = RobertaConfig(output_config_file)
        model = RobertaForMultipleChoice(config)
        model.load_state_dict(torch.load(output_model_file))
    else:
        model = RobertaForMultipleChoice.from_pretrained(args.RoBerta_model)
        load_output_model(model, output_model_file)
        # model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = load_data(args, is_training=False)
        eval_features = convert_examples_to_features(
            eval_examples, tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_results = []
        for input_ids, input_mask, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # tmp_eval_loss = model(input_ids, input_mask, label_ids)
                tmp_eval_loss = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
                logits = model(input_ids=input_ids, attention_mask=input_mask)
            #print(logits)

            # logits = logits.detach().cpu().numpy()
            logits = logits.numpy()
            eval_results.append(logits)
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += tmp_eval_loss.loss().mean()

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
                result['loss'] = tr_loss / nb_tr_steps
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
