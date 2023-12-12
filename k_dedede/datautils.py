from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Optional
from typing_extensions import Self
import os 
from os import path
import pickle
from copy import deepcopy
import wget
import shutil

import torch 
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from k_dedede.env import Env
from k_dedede.glue.download_glue_data import download_and_extract, download_diagnostic
from k_dedede.glue.src.tasks import MultiNLITask
from k_dedede.fdistill.utils import LegacySeq2SeqDataset, SortishSampler



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
    def get_train_test_split():
        raise NotImplementedError
    def collate_fn(self):
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


@dataclass 
class Batch:
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None, 
    labels: Optional[torch.Tensor] = None, 
    decoder_attention_mask: Optional[torch.Tensor]= None



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

    def __len__(self) -> int:
        raise NotImplementedError 

    def __getitem__(self, index:int) -> Any:
        raise NotImplementedError
        
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
    
    def get_train_test_split(tokenizer: PreTrainedTokenizer) -> Tuple[Self, Self, Self]:
        train = MNLI(tokenizer, split="train", load_full_data=True)
        val = MNLI(tokenizer, split="val", load_full_data=False)
        test = MNLI(tokenizer, split="test", load_full_data=False)
        val.data = deepcopy(train.val_data_text)
        test.data = deepcopy(train.test_data_text)
        train.cleanup()
        return (train, val, test)        
    
    def collate_fn(self):
        return super().collate_fn()



class DART(DatasetBase):

    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 split:str="train", 
                 load_full_data: bool=True):
        """
        load_full_data: If false, doesn't load anything
        """
        super().__init__(tokenizer, split=split)
        legacy_args = {}
        legacy_args["tokenizer"] = tokenizer         
        legacy_args["data_dir"] = ""
        legacy_args["max_source_length"] = None
        legacy_args["max_target_length"] = None

        self.data_folder = Env.data 
        self.cache_folder = Env.dataset_cache
        rebuild_dataset = Env.rebuild_dataset 
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        self.dataset_path = path.join(self.data_folder, "DART")
        self.cache_file = path.join(self.cache_folder, "dart.pkl")

        self.dataset = None

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

    def __len__(self) -> int:
        assert self.dataset is not None, "Dataset not built yet. Please try rerunning the script with the rebuild_dataset flag set."
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert self.dataset is not None, "Dataset not built yet. Please try rerunning the script with the rebuild_dataset flag set."
        item =  self.dataset[idx] 
        return item

    def rebuild(self) -> Self:
        os.makedirs(self.dataset_path, exist_ok=True) 
        # remove all files from self.dataset_path
        for filename in os.listdir(self.dataset_path):
            file_path = os.path.join(self.dataset_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #download all files
        fname = wget.download(
            "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-train.json", 
            out=path.join(self.dataset_path, "train.json"))
        print(f"\n{fname} downloaded")
        fname = wget.download(
            "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-dev.json", 
            out=path.join(self.dataset_path, "val.json"))
        print(f"\n{fname} downloaded")
        fname = wget.download(
            "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-test.json", 
            out=path.join(self.dataset_path, "test.json"))
        print(f"\n{fname} downloaded")
        n_observations_per_split = {
            "train": -1, # -1 means use all
            "val": 500,
            "test": -1,
        }
        legacy_args = {}
        legacy_args["tokenizer"] = self.tokenizer         
        legacy_args["data_dir"] = self.dataset_path
        legacy_args["type_path"] = self.split
        legacy_args["max_source_length"] = 384
        legacy_args["max_target_length"] = 150
        legacy_args["prefix"] = ""
        self.dataset = LegacySeq2SeqDataset(**legacy_args)
        legacy_args["type_path"] = "val"
        self.val_split = LegacySeq2SeqDataset(**legacy_args)
        legacy_args["type_path"] = "test"
        self.test_split = LegacySeq2SeqDataset(**legacy_args)
        return self

    def cleanup(self) -> Self:
        if self.split == "train":
            del self.val_split
            del self.test_split
        elif "val" in self.split:
            self.dataset = self.val_split
            del self.test_split
        else:
            self.dataset = self.test_split
            del self.val_split
        return self

    def change_split(self, split:str) -> Self:
        self.split = split
        return self
    
    def get_train_test_split(tokenizer: PreTrainedTokenizer) -> Tuple[Self, Self, Self]:
        train = DART(tokenizer, split="train", load_full_data=True)
        val = DART(tokenizer, split="val", load_full_data=False)
        test = DART(tokenizer, split="test", load_full_data=False)
        val.dataset = deepcopy(train.val_split)
        test.dataset = deepcopy(train.test_split)
        train.cleanup()
        return (train, val, test)        
    
    def collate_fn(self, batch: Dict[str, torch.Tensor]) -> Batch:
        collated = self.dataset.collate_fn(batch)
        return Batch(input_ids=collated["input_ids"], 
                     attention_mask=collated["attention_mask"], 
                     labels=collated["labels"], 
                     decoder_attention_mask=collated["target_attention_mask"])

if __name__ == "__main__":
    Env.setup("config/general.yaml")
    tok = AutoTokenizer.from_pretrained(Env.tfmr)
    print("loading train")
    train = DART(tok)

    train, val, test = DART.get_train_test_split(tok)
    train_loader = DataLoader(train, batch_size=32, shuffle=True,
        num_workers=0, collate_fn=train.collate_fn)
    pass
    pass