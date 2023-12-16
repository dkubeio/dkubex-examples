d3x ray job submit --submission-id llamaft-9 \
           --runtime-env-json '{"pip":["mlflow"]}' \
           --working-dir $PWD \
           -- python finetuning.py --use_peft --peft_method lora \
                                   --model_name $HOME/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/ \
                                   --batch_size_training 1 \
                                   --num_epochs 1 \
                                   --experiment_name llama27bft_25_oct_23 --run_name finetuning \
                                   --dist_checkpoint_root_folder checkpoints/docs  \
                                   --output_dir $HOME/PEFT/docs/model_v1_25-oct-23  \
                                   --dataset docs_dataset --data_path  /home/sagar-suman/Downloads/dkubex-fm/tools/processed_for_finetuning \
                                   --run_validation False --batch_size 1 \
				   --num_workers 4 --enable_fsdp \
				   --pure_bf16
				   #--num_workers 1 \
                                   #--quantization --use_fp16 \ . int8 quantization is not supported with fsdp
