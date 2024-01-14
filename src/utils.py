import os
import json
import time
import logging
import threading
from functools import wraps
from typing import Optional, Literal, Any
from dataclasses import field, dataclass, asdict

import torch
from transformers import TrainingArguments as TransformerTrainingArguments, TrainerCallback, HfArgumentParser


logging.basicConfig(level=logging.INFO)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Increase timeout to prevent wandb errors
os.environ["WANDB__SERVICE_WAIT"] = "300"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
# Setting to parallelism to true
os.environ["TOKENIZERS_PARALLELISM"] = "True"


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32
}


@dataclass
class ModelArguments:
    dtype: Optional[Literal["bf16", "f16", "f32"]] = field(default="bf16")
    load_in_4bit: Optional[bool] = field(default=False)
    load_in_8bit: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="adept/fuyu-8b")


@dataclass
class DataArguments:
    train_samples: Optional[int] = field(default=256)
    eval_samples: Optional[int] = field(default=16)
    dataset_json: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(TransformerTrainingArguments):
    algo: Literal["dpo", "sft"] = "dpo"
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_exclude: list[str] = field(default_factory=list)
    dpo_beta: float = 0.1
    wandb_project: Optional[str] = field(default=None)


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def get_arg_dicts(arg_file: str) -> list[dict[str, Any]]:
    with open(arg_file, "r") as f:
        arg_dicts = json.load(f)
    return [
        arg_dicts["model_args"],
        arg_dicts["data_args"],
        arg_dicts["training_args"]
    ]

def dump_arg_dicts(arg_dicts: dict[str, dict[str, Any]], output_dir: str, filename: str = "arg_dict.json"):
    arg_dict = {name: asdict(arg) for name, arg in arg_dicts.items()}

    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(arg_dict, f, indent=2)


def get_args(arg_file: str) -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    modal_arg_dict, data_arg_dict, training_arg_dict = get_arg_dicts(arg_file)

    # TODO: figure out a way to not parse separately and preserve types
    model_arg_parser = HfArgumentParser(ModelArguments)
    model_args: ModelArguments = model_arg_parser.parse_dict(modal_arg_dict)[0]
    data_arg_parser = HfArgumentParser(DataArguments)
    data_args: DataArguments = data_arg_parser.parse_dict(data_arg_dict)[0]
    training_arg_parser = HfArgumentParser(TrainingArguments)
    training_args: TrainingArguments = training_arg_parser.parse_dict(training_arg_dict)[0]
    return model_args, data_args, training_args


def find_all_linear_names(model, exclude: list[str]):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(keyword in name for keyword in exclude):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)


def format_messages(batch: list[list[str]]) -> list[list[dict[str, str]]]:
    return [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": m
    } for i, m in enumerate(b)] for b in batch]


def throttle(lock: threading.Lock, rqi: int, last_requests: list[float], interval: int = 60) -> None:
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      nonlocal last_requests
      exceeds_rpm = True
      while exceeds_rpm:
        with lock:
          # Remove timestamps older than 60 seconds
          now = time.time()
          last_requests = [req for req in last_requests if now - req < interval]
          # If the number of requests in the last minute is less than the limit, send a new request
          if len(last_requests) < rqi:
            last_requests.append(now)
            exceeds_rpm = False
          else:
            # Otherwise, wait for some time and try again
            minimum = min([now - req for req in last_requests])
            time.sleep(minimum)
      
      return func(*args, **kwargs)
    return wrapper
  return decorator