from utils import install_dependencies, set_seed, print_items_from_dict
set_seed(42)
import json
import argparse
from model import T5FineTuner
import pytorch_lightning as pl
from transformers import T5Tokenizer
from dataset import wikisql, get_dataset
from callback import LoggingCallback, logger
from dataset import get_dataset



import json
import argparse

with open('config.json') as f:
  args_dict = json.load(f)
args = argparse.Namespace(**args_dict)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
dataset = get_dataset(tokenizer=tokenizer, type_path="train", num_samples=100, args=args, sql2txt=False)