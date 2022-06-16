#!/bin/bash
python -m qa.run_bert --bert_model bert-base-uncased --do_lower_case --output_dir=C:\Users\Stav\Desktop\reading_comprehension-master_jon\reading_comprehension-master\output dir --finetune --run-yev --do_train --do_eval --max_seq_length 512 --gradient_accumulation_steps 10 --num_train_epochs 1 --eval_batch_size 16 --max_batches 50
