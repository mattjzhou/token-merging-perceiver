All experiments were done using the huggingface library
GLUE experiments:
    python run_glue.py --model_name_or_path deepmind/language-perceiver --do_train --do_eval --do_predict --task_name <TASK-NAME> --output_dir <DIR> --num_train_epochs 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --learning_rate .00002 --weight_decay .01 --warmup_steps 200 --save_steps .2 --max_seq_length 2048
    python language_tome.py --model_path <DIR>/<CHECKPOINT> --task_name <TASK-NAME> --r 8 --output_dir <output_dir> --num_train_epochs 5 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .00002 --weight_decay .01 --save_steps .2

FashionMNIST experiments
    python run_image_classification.py --model_name_or_path deepmind/vision-perceiver-conv --do_train --do_eval --do_predict --dataset_name fashion_mnist --output_dir <DIR> --ignore_mismatched_sizes --num_train_epochs 2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .0001 --weight_decay .01 --warmup_steps 200 --save_steps .2
    python image_tome.py --model_path <DIR>/<CHECKPOINT> --dataset_name fashion_mnist --r 16 --output_dir <output_dir> --num_train_epochs 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --learning_rate .00005 --weight_decay .01 --save_steps .2