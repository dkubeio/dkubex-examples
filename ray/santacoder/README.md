
# Ray Training 
Fine-tune SantaCoder on Code and Text Generation datasets. For example on new programming languages from The Stack dataset, or on a code-to-text dataset like GitHub-Jupyter. SantaCoder is a 1B parameters model pre-trained on Python, Java & JavaScript, we suggest fine-tuning on programming languages close to them, otherwise, the model might not converge well.
https://github.com/loubnabnl/santacoder-finetuning

- login
```
$ pip install wandb==0.13.1 huggingface-hub
$ huggingface-cli login
$ wandb login
```
- Create a ray cluster 
```
 $ d3x ray create -n gpu --cpu=8 --memory=64 --hcpu=8 --hmemory=64 --hgpu=1 --gpu=1  -i rayproject/ray-ml:2.7.0.d37759-py310-gpu  -v 2.7.0
```
- Activate ray cluster
```
 $ d3x ray activate
```
- Run the training
```
$ d3x ray job submit --working-dir $PWD --runtime-env-json='{"pip": "./requirements.txt"}' -- python trainray.py \
	--model_path="bigcode/santacoder" \
        --dataset_name="bigcode/the-stack-dedup" \
        --subset="data/shell" \
        --data_column "content" \
        --split="train" \
        --seq_length 2048 \
        --max_steps 30000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --eval_freq 3000 \
        --save_freq 3000 \
        --log_freq 1 \
        --num_workers=4 \
	--no_fp16 \
        --use-gpu

```
## Testing with smaller data
```
$ d3x ray create -n test   --hcpu 8  --hmemory 64   -i rayproject/ray-ml:2.7.0.d37759-py310-gpu  -v 2.7.0
$ d3x ray activate test
$ d3x ray job submit --working-dir $PWD --runtime-env-json='{"pip": "./requirements.txt"}' -- python trainray.py --model_path=bigcode/santacoder --dataset_name=bigcode/the-stack-dedup --subset=data/shell --data_column content --split=train --seq_length 2048 --max_steps 1 --batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --num_warmup_steps 1 --eval_freq 1 --save_freq 1 --log_freq 1 --num_workers=1 --no_fp16  --size_valid_set 1
```
