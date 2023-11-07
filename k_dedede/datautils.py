from typing import Any, List, Tuple
from typing_extensions import Self
import os 
from os import path
import pickle
from copy import deepcopy


import torch 
from torch import nn 
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from k_dedede.env import Env
from k_dedede.glue.download_glue_data import download_and_extract, download_diagnostic
from k_dedede.glue.src.tasks import MultiNLITask




class DatasetBase(Dataset):
    def __init__(self, tokenizer:PreTrainedTokenizer, name:str=None, split:str="train"):
        self.tokenizer = tokenizer
        self.name = name 
        self.split = split
    def __len__(self) -> int:
        raise NotImplementedError 
    def __getitem__(self, index:int) -> Any:
        raise NotImplementedError
    def rebuild(self):
        raise NotImplementedError("rebuild not implemented")
    def cleanup(self):
        raise NotImplementedError
    def get_train_test_split(self):
        raise NotImplementedError
    def to_cache(self, filename:str) -> Self:
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)
        return self

    def from_cache(self, filename:str) -> Self:
        with open(filename, "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        return self

class MNLI(DatasetBase, MultiNLITask):
    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 split:str="train", 
                 load_full_data: bool=True):
        """
        load_full_data: If false, doesn't load anything
        """
        super().__init__(tokenizer, split=split)
        super(DatasetBase, self).__init__()
        # pull from Env
        self.data_folder = Env.data 
        self.cache_folder = Env.dataset_cache
        rebuild_dataset = Env.rebuild_dataset
        # continue setup
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)
        self.dataset_path = path.join(self.data_folder, "MNLI")
        self.cache_file = path.join(self.cache_folder, "mnli.pkl")
        if load_full_data:
            try:
                # load from cache
                if rebuild_dataset:
                    print("Rebuilding dataset...")
                    self.rebuild()
                    print("Rebuild success!")
                else: 
                    print("Trying to load from cache...")
                    self.from_cache(self.cache_file)
                    print("Loaded from cache!")
            except:
                # force rebuild dataset
                print("Dataset not in cache. Rebuilding...")
                self.rebuild()
                print("Rebuild success!")
            self.to_cache(self.cache_file)
            print(f"Dataset saved to {self.cache_file}.")

    def rebuild(self) -> Self:
        download_and_extract("MNLI", self.data_folder)
        download_diagnostic(self.dataset_path)
        os.rename(path.join(self.dataset_path, "diagnostic", "diagnostic.tsv"), 
                    path.join(self.dataset_path, "diagnostic.tsv"))
        self.load_data(self.dataset_path, max_seq_len=128)
        return self

    def cleanup(self) -> Self:
        if self.split == "train":
            self.data = self.train_data_text
            del self.test_data_text
            del self.val_data_text
        elif "val" in self.split:
            self.data = self.val_data_text
            del self.test_data_text
            del self.train_data_text
        else:
            self.data = self.test_data_text
            del self.train_data_text
            del self.val_data_text
        return self
    def change_split(self, split:str) -> Self:
        self.split = split
        return self
    
    def get_train_test_split(self) -> Tuple[Self, Self, Self]:
        val = MNLI(self.tokenizer, split="val", load_full_data=False)
        test = MNLI(self.tokenizer, split="test", load_full_data=False)
        val.data = deepcopy(self.val_data_text)
        test.data = deepcopy(self.test_data_text)
        self.cleanup()
        return (self, val, test)        



if __name__ == "__main__":
    Env.setup("config/general.yaml")
    tok = AutoTokenizer.from_pretrained("roberta-base")
    print("loading train")
    train = MNLI(tok)
    train, val, test = train.get_train_test_split()