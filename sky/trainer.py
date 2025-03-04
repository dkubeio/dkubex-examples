import argparse
import json
import os
from os import path
import sys
from functools import partial

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
#from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from trl import SFTTrainer

from transformers.utils import logging

logger = logging.get_logger("dkubex-llmtrainer")

os.environ["WANDB_DISABLED"] = "true"

import ray
logger.info(f"================ RAY VERSION {ray.__version__} ================")

def datafunc():
    import wget
    import json
    import os
    import pandas as pd

    if os.path.exists("data.json") == False:
        wget.download("https://github.com/gururise/AlpacaDataCleaned/raw/main/alpaca_data_cleaned.json", "./data.json")

    if os.path.exists("data_purged.json") == False:
        with open("data.json", "r+") as f:
            data = json.load(f)

            with open("data_purged.json", "w+") as f:
                purged = data[:100]
                json.dump(purged, f)

    with open("data_purged.json") as f:
        json_data = json.load(f)

    PROMPT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    textcol = []
    #df = pd.read_json(json_data)
    df = pd.json_normalize(json_data)
    for _, row in df.iterrows():
        prompt = PROMPT
        instruction = row["instruction"]
        input_text = row["input"]
        output_text = row["output"]

        if len(input_text.strip()) == 0:
            text = f"{prompt} \n ### Instruction: \n {instruction} \n### Response: \n {output_text} \n"
        else:
            text = f"{prompt} \n ### Instruction: \n {instruction} \n### Input: \n {input_text} \n### Response: \n {output_text} \n"
        textcol.append(text)

    df.loc[:, "text"] = textcol
    print(df.head())

    df.to_csv("training_data.csv")

#def trainfunc(model, token, epochs, store):
def trainfunc(train_config):
    datafunc()

    model = train_config["model"]
    token = train_config["token"]
    epochs = train_config["epochs"]
    store = train_config["store"]

    training_data_csv = "./training_data.csv"
    train_data = pd.read_csv(training_data_csv)
    train_data = Dataset.from_pandas(train_data)

    #model_path = "./llmodels/"
    token = "$your huggingface token$"
    model_path = model

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_auth_token=token,
        trust_remote_code=True,
    )

    #MAK - can I leave this to default which is per model ?
    tokenizer.model_max_length = 2048

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(
        model_path,
        use_auth_token=token,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        use_auth_token=token,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": Accelerator().process_index} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    #MAK - Do I need this line ?
    #model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'],
    )
    bmodel = model
    model = get_peft_model(model, peft_config)


    block_size = tokenizer.model_max_length
    if block_size > 1024:
        block_size = 1024

    block_size = 100
    batch_size = 2
    logging_steps = int(0.2 * len(train_data) / batch_size)

    if logging_steps == 0:
        logging_steps = 1

    mlflow_runname = train_config['mlflow_runname']
    outputdir = path.join(store, mlflow_runname)
    #outputdir = f"{store}/tunedllm"
    import shutil
    shutil.rmtree(outputdir, ignore_errors=True)

    training_args = dict(
        output_dir=outputdir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        evaluation_strategy="no",
        logging_steps=logging_steps,
        save_total_limit=1,
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        report_to="mlflow",
        auto_find_batch_size=False,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=False,
        push_to_hub=False,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
    )

    args = TrainingArguments(**training_args)

    trainer_args = dict(
        args=args,
        model=model,
    )

    trainer = SFTTrainer(
        **trainer_args,
        train_dataset=train_data,
        eval_dataset=None,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=block_size,
        tokenizer=tokenizer,
        packing=True,
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    mlflow_runname = train_config['mlflow_runname']
    mlflow_expname = train_config['mlflow_expname']

    import mlflow
    mlflow.set_experiment(experiment_name=mlflow_expname)
    with mlflow.start_run(run_name=mlflow_runname) as root_run:
        trainer.train()

    #trainer.train()


    logger.info("Finished training, saving model...")

    #trainer.save_model(outputdir)
    model.save_pretrained(outputdir)

    # merge the peft adapter and base model as one
    """
    bmodel = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_auth_token=token,
        )

    """

    from peft import PeftModel
    pmodel = PeftModel.from_pretrained(bmodel, outputdir)

    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_auth_token=token,
    )
    """

    model = pmodel.merge_and_unload()

    logger.info("Saving target model...")
    pmodel.save_pretrained(f"{outputdir}/merged")
    tokenizer.save_pretrained(f"{outputdir}/merged")

import mlflow
class TunedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, tuned_model):
        self.tuned_model = tuned_model

    def predict(self, context, model_input):
        return {}

def trainer(model="meta-llama/Llama-2-7b-chat-hf", token="", nworkers=1, epochs=1, store="/tmp/"):
    import ray
    from ray.train.huggingface import AccelerateTrainer
    from ray.air.config import ScalingConfig, RunConfig
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    import mlflow
    import string
    import random

    address = os.getenv("RAY_ADDRESS")
    ray.init(address=address)

    user = os.environ.get("USER", "anonymous")


    name = "llama27b-ft-" + ''.join(random.choices(string.ascii_lowercase +
                                 string.digits, k=6))


    tags = { "mlflow.user" : user, "trainerid": name }

    run_cfg = RunConfig(
                name=name,
                callbacks=[
                    MLflowLoggerCallback(
                        tags=tags,
                        experiment_name="llama27b-ft",
                        save_artifact=True,
                    )
                ],)


    run_cfg = RunConfig(name=name)

    trainer = AccelerateTrainer(
        trainfunc,
        train_loop_config={"model": model, "token": token, "epochs": 1, "mlflow_runname": name, "mlflow_expname": "llama27b-ft", "store": store},
        accelerate_config={},
        scaling_config=ScalingConfig(
            num_workers=nworkers,
            use_gpu=True
        ),
        run_config=run_cfg)


    result = trainer.fit()


    filter_string=f"attributes.run_name = '{name}'"
    df = mlflow.search_runs(experiment_names=["llama27b-ft"], filter_string=filter_string)
    run_id = df.loc[0,'run_id']
    
    with mlflow.start_run(run_id=run_id) as run:
        spath = path.join(store, name)
        mlflow.log_artifacts(spath, artifact_path="tuned_model")

        tmodel = TunedModel("tuned_model")
        mlflow.pyfunc.log_model("llama27bchat_tunedmodel", python_model=tmodel, artifacts={"tuned_model" : mlflow.get_artifact_uri(artifact_path="tuned_model")}, registered_model_name="llam27bchat_finetuned")


    #datafunc()
    #trainfunc(model, token, epochs, store)

if __name__ == '__main__':
    import fire
    fire.Fire(trainer)
