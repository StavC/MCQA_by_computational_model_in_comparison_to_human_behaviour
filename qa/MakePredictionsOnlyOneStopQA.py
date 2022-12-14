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

"""
Most of the code was taken from here :  https://github.com/rodgzilla/pytorch-pretrained-BERT/tree/multiple-choice-code/examples

John's edited the code first and then then Stav and Nurit changed it to run on RoBerta and not BERT

this code should just make predictions on OneStopQA and save a CSV file of the predictions.

Need a trained model on Race + OneStopQA or Race.


t
Parmas just to run predictions on OneStopQA
--RoBerta_model=roberta-base --output_dir=/home/cohnstav/ModelWeights/resultsOnOneSTOPQATest --do_eval --run-yev

to run just on race
--RoBerta_model=roberta-base --output_dir=/home/cohnstav/ModelWeights/resultsOnOneSTOPQATest --do_eval
"""


import argparse
import csv
import pandas as pd
import logging
import os
import random
from io import open

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

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
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            # TODO: Stav Thinks we can change the truncate function to cut more context and not answers

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


def load_data2(args, is_training=True):  # use this when running from home
    if args.run_yev:
        print("Using yev data")
        return read_onestop(is_training=is_training)
    elif (not (args.run_yev)) and (is_training == True):
        return read_race_examples(Path(__file__).parent.parent / 'data' / 'RACE' / 'train' / 'combined',
                                  is_training=is_training)
    else:
        return read_race_examples(Path(__file__).parent.parent / 'data' / 'RACE' / 'dev' / 'combined',
                                  is_training=is_training)


def load_data(args, is_training=False):  # use this when running through tmux
    if args.run_yev:
        print("Using yev data")
        return read_onestop(is_training=is_training)
    else:
        return read_race_examples(Path('/tmp/pycharm_project_972/reading_comprehension-master/data/RACE/test/combined'),
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
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        # default=5e-5,
                        default=0.000008,  # 0.00001 for roberta on race
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=15,
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
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), ))

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)

    # Prepare model
    model = RobertaForMultipleChoice.from_pretrained('roberta-base')

    # load the model that was trained on Roberta and then OneSTOPQA
    load_output_model(model, os.path.join(args.output_dir,
                                          '/home/cohnstav/ModelWeights/Fold4Weights/9acc:0.7638888888888888OneStopQAFold4.pt'))  # fold 1 or original split

    model.to(device)
    print(
        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(n_gpu)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # make predictions
    eval_examples = load_data(args, is_training=False)

    dataAndPrediction = []
    for row in eval_examples:

        eval_features = convert_examples_to_features(
            [row], tokenizer, args.max_seq_length, True)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label)
        # Run prediction for full data

        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

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
                logitsCopy = logits.copy()

            logits = logits.logits.detach().cpu().numpy()
            eval_results.append(logits)
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            ###stav addons for analyze
            # print(row)
            """
            print(row.id)
            print(row.id.paragraph_id.article_id)
            print(row.id.paragraph_id.paragraph_id)
            print(row.id.paragraph_id.level)
            print(row.id.question)
            print('@@@@@@@@@@@')
            print(logits)
            print(label_ids)"""
            m = torch.nn.Softmax(dim=1)
            softmax = m(logitsCopy.logits)
            predicted_label = np.argmax(logits)
            dataAndPrediction.append(['OneStopQA', row.id.paragraph_id.article_id, row.id.paragraph_id.paragraph_id,
                                      row.id.paragraph_id.level, row.id.question, row.id, row.context_sentence,
                                      row.start_ending, row.endings[0], row.endings[1], row.endings[2],
                                      row.endings[3], row.label, logits, softmax, predicted_label])

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += tmp_eval_loss.loss.mean()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
    predictionDF = pd.DataFrame(data=dataAndPrediction,
                                columns=['Dataset', 'article_id', 'paragraph_id', 'level', 'questionNumber',
                                         'QuestionId', 'context', 'question', 'ending0', 'ending1', 'ending2',
                                         'ending3', 'label', 'logits', 'softmax', 'predicted_label'])

    # predictionDF.to_csv(r'PredictionsOnRaceTest.csv',index=False)
    predictionDF.to_csv(r'PredictionsOnOneStopQAFold1.csv', index=False)
    output_eval_pickle = os.path.join(args.output_dir, "eval_results.pickle")
    torch.save(eval_results, output_eval_pickle)
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              }
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
