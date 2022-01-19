from transformers import T5Tokenizer
from torch.utils.data import Dataset
import pandas as pd


class wikisql(Dataset):
    def __init__(self, type_path: str,
                 input_length: int,
                 output_length: int,
                 num_samples: int = None,
                 tokenizer=T5Tokenizer.from_pretrained('google/mt5-small'),
                 sql2txt: bool = False) -> None:

        if type_path == "train":
            self.dataset = pd.read_csv("data/train_df_pl.csv")
            self.dataset = self.dataset.loc[0:50, :]
        elif type_path == "validation":
            self.dataset = pd.read_csv("data/dev_df_pl.csv")
        elif type_path == "test":
            self.dataset = pd.read_csv("data/test_df_pl.csv")

        """if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))"""
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.sql2txt = sql2txt

    def __len__(self) -> int:
        return self.dataset.shape[0]

    def clean_text(self, text: str) -> str:
        return text.replace('\n', '').replace('``', '').replace('"', '')

    def convert_to_features(self, example_batch):
        if self.sql2txt:
            # sql to text
            input_ = "translate SQL to English: " + self.clean_text(example_batch['sql']['human_readable'])
            target_ = self.clean_text(example_batch['question'])
        else:
            # text to sql
            input_ = "translate English to SQL: " + self.clean_text(example_batch['question']) + self.clean_text(example_batch["header"])
            print(input_)
            sql = example_batch["sql"]
            human_readable = eval(sql)["human_readable"]
            target_ = self.clean_text(human_readable)

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                  padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                   padding='max_length', truncation=True, return_tensors="pt")

        return source, targets

    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset.loc[index, :])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


def get_dataset(tokenizer, type_path: str, num_samples: int, args, sql2txt) -> wikisql:
    return wikisql(type_path=type_path,
                   num_samples=num_samples,
                   input_length=args.max_input_length,
                   output_length=args.max_output_length,
                   sql2txt=sql2txt)
