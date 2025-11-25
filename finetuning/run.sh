d3x ray job submit --submission-id llamaft-8 \
           --runtime-env-json '{"pip":["mlflow"]}' \
           --working-dir $PWD \
           -- python finetuning.py --use_peft --peft_method lora \
                                   --model_name $HOME/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/08751db2aca9bf2f7f80d2e516117a53d7450235/ \
                                   --batch_size_training 4 \
                                   --num_epochs 3 \
                                   --experiment_name llama27bft_12_oct_23 --run_name finetuning \
                                   --dist_checkpoint_root_folder checkpoints/docs  \
                                   --output_dir $HOME/PEFT/docs/model_v1_12-oct-23  \
                                   --dataset docs_dataset --data_path $HOME/chunks_finetuning_aggregate \
                                   --run_validation False --batch_size 1 \
				   --num_workers 3 --enable_fsdp \
				   --pure_bf16
				   #--num_workers 1 \
                                   #--quantization --use_fp16 \ . int8 quantization is not supported with fsdp
