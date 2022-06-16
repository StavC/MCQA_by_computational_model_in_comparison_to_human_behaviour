#!/bin/bash
python -m qa.run_bidaf --bert_model bert-base-uncased --do_lower_case --output_dir=$HOME/models/race  --do_train --do_eval --max_seq_length 512 --gradient_accumulation_steps 3 --num_train_epochs 3 --eval_batch_size 3   --d 100 --train_batch_size 12
