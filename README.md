# Efficient Machine Learning Final Project: Token Merging Perceiver
All experiments were done using the huggingface library
## GLUE Experiments

Finetune on GLUE:

    python run_glue.py --model_name_or_path deepmind/language-perceiver --do_train --do_eval --do_predict --task_name <TASK-NAME> --output_dir <DIR> --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate .00002 --weight_decay .01 --warmup_steps 200 --save_steps .2 --max_seq_length 2048

Token Merging & Retraining:

    python language_tome.py --model_path <DIR>/<CHECKPOINT> --task_name <TASK-NAME> --r 8 --output_dir <output_dir> --num_train_epochs 5 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .00002 --weight_decay .01 --save_steps .2

## FashionMNIST Experiments

Finetune on FashionMNIST:

    python run_image_classification.py --model_name_or_path deepmind/vision-perceiver-conv --do_train --do_eval --do_predict --dataset_name fashion_mnist --output_dir <DIR> --ignore_mismatched_sizes --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .0001 --weight_decay .01 --warmup_steps 200 --save_steps .2

Token Merging & Retraining:

    python image_tome.py --model_path <DIR>/<CHECKPOINT> --dataset_name fashion_mnist --r 16 --output_dir <output_dir> --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .00005 --weight_decay .01 --save_steps .2