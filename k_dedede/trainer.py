from dataclasses import dataclass
import os 
from os import path 
from typing import List, Dict, Tuple, Union, Optional, Callable
from typing_extensions import Self
from tqdm import tqdm, trange
from datetime import datetime
import pickle 

import torch 
from torch import nn 
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSeq2SeqLM,  AutoTokenizer, PreTrainedTokenizer
from accelerate import Accelerator


from k_dedede.taskmodel import TaskModel, BartTaskModel, RobertaTaskModel
from k_dedede.datautils import DatasetBase, DART, Batch
from k_dedede.env import Env

@dataclass
class TrainerMetaData:
    step: int
    epoch: int
    best_metric: int

class BaseTrainer:
    def __init__(self, 
                 model: TaskModel, 
                 tokenizer: PreTrainedTokenizer,  
                 dataset_class: Callable, 
                 optimizer: Optimizer, 
                 scheduler: Optional[Callable] = None):
        self.accelerator = Accelerator()

        self.model = model.to(self.accelerator.device)
        self.optim = optimizer
        self.tokenizer = tokenizer 
        self.scheduler = scheduler if scheduler is not None else ExponentialLR(self.optim, gamma=0.99)


        self.epochs = Env.epochs 
        self.batch_size = Env.batch_size
        self.lr = Env.lr
        self.device = Env.device 
        self.eval_every_steps = Env.eval_every_steps
        self.best_metric = None

        self.exp_name = datetime.now().strftime("%m-%d-%Y--%H:%M:%S") if Env.exp_name is None else Env.exp_name
        self.save_dir = path.join(Env.output_dir, self.exp_name)

        os.makedirs(self.save_dir, exist_ok=True)
        self.trainset, self.valset, self.testset = dataset_class.get_train_test_split(tokenizer)
        self.train_loader = self.get_dataloader(self.trainset)
        self.val_loader = self.get_dataloader(self.valset) if self.valset is not None else None
        self.test_loader = self.get_dataloader(self.testset) if self.testset is not None else None

        self.model, self.optim, self.train_loader = self.accelerator.prepare(
            self.model, self.optim, self.train_loader 
        )
        self.accelerator.register_for_checkpointing(self.scheduler)
        self.val_loader = self.accelerator.prepare(self.val_loader) if self.val_loader is not None else None
        self.test_loader = self.accelerator.prepare(self.test_loader) if self.test_loader is not None else None

        try: # load previous train state
        # if True:
            print(f"Trying to resume experiment {Env.exp_name}") if Env.exp_name is not None else None
            trainermeta = self.load_state(self.save_dir)
            
            self.best_metric = trainermeta.best_metric
            self.step = trainermeta.step 
            self.current_epoch = trainermeta.epoch
            
        except:
            if Env.exp_name is not None:
                print(f"Failed to resume experiment {Env.exp_name}.") 
            print(f"Starting new experiment.")

        
            self.step = 0
            self.current_epoch = 0


    def run_loop(self):
        end_epoch = self.current_epoch + self.epochs
        for epoch in range(self.current_epoch, end_epoch):
            first_loaded_step = self.step
            tbar = tqdm(enumerate(self.train_loader, start=self.step), 
                         initial=self.step, 
                         desc=f"Training epoch {epoch + 1}/{end_epoch}", 
                         total=len(self.train_loader), 
                         dynamic_ncols=True)

            for i, batch in tbar:
                batch = batch.to(self.device)
                loss = self.train_step(batch)
                
                self.step = i
                self.current_epoch = epoch 

                tbar.set_postfix({"loss": loss})
                

                if self.step != first_loaded_step and \
                    self.step % self.eval_every_steps == 0 and \
                    self.val_loader is not None:
                    optimizing_metric = 0
                    with torch.no_grad():
                        for v_batch in tqdm(self.val_loader, 
                                            desc=f"Validation", 
                                            ncols=80):
                            v_batch = v_batch.to(self.device)
                            opt_met, maximize = self.val_step(batch)
                            optimizing_metric += opt_met*(1 / len(self.val_loader))
                    self.save_best_model(model, optimizing_metric, maximize=maximize)
                    self.save_state(self.save_dir)



            self.scheduler.step() if self.scheduler is not None else None
            self.step = 0

    def train_step(self, batch: Batch) -> float:
        raise NotImplementedError

    def val_step(self, batch: Batch) -> Tuple[float, bool]: 
        """
        returns optimization metric as well as whether or not to maximize
        """
        raise NotImplementedError 
                
    def get_dataloader(self, dataset: DatasetBase) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    def get_optimizer_scheduler(self, 
                                optim: str, 
                                model: nn.Module, 
                                sched: Optional[str] = None) -> Tuple[Optimizer, Optional[Callable]]:
        if optim == "adamw": 
            optimizer = AdamW(model.parameters(), lr=self.lr)
        elif optim == "adam":
            optimizer = Adam(model.parameters(), lr=self.lr)
        elif optim == "sgd":
            optimizer = SGD(model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError 
        
        if sched is None:
            return optim, None 
        return optimizer, ExponentialLR(optimizer, 0.99)

    def save_best_model(self, model: TaskModel, 
                        optimizing_metric: float,
                        maximize: bool = False) -> TaskModel:
        """
        saves best model based on the optimizing metric
        """
        if self.best_metric is None:
            self.best_metric = - float("inf") if maximize else float("inf")
        flip = -1 if maximize else 1
        best_save_path = path.join(self.save_dir, "best") 
        if flip * optimizing_metric < flip * self.best_metric:
            self.accelerator.unwrap_model(model).save(best_save_path)
            self.best_metric = optimizing_metric
            with open(path.join(best_save_path, "trainstate.txt"), "a") as f:
                f.write(f"Epoch: {self.current_epoch}, Step: {self.step}, Optimizing Metric: {optimizing_metric:.4f}, Maximize: {'True' if maximize else 'False'}\n")
                
    def save_state(self, fpath:str):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save(self.save_dir)
        # trainstate = TrainerState(
        #     model=self.model
        #     tokenizer=self.tokenizer, 
        #     optim_state_dict=self.optim.state_dict(), 
        #     step=self.step, 
        #     epoch=self.current_epoch, 
        #     best_metric=self.best_metric
        # )
        # with open(path.join(fpath, "trainstate.pt"), "wb") as f:
        #     pickle.dump(trainstate, f)
        
        self.accelerator.save_state(path.join(fpath, "checkpoints"))
        # return trainstate
        trainermeta = TrainerMetaData(
            step=self.step, 
            epoch=self.current_epoch, 
            best_metric=self.best_metric
        )
        with open(path.join(fpath, "trainer-metadata.data"), "wb") as f:
            pickle.dump(trainermeta, f)



    
    def load_state(self, fpath: str) -> TrainerMetaData:
        # with open(path.join(fpath, "trainstate.pt"), "rb") as f:
        #     trainerstate = pickle.load(f)
        
        # trainerstate = torch.load(path.join(fpath, "trainstate.pt"))
        # self.model = self.model.load(fpath)
        # return trainerstate
        self.accelerator.load_state(path.join(fpath, "checkpoints"))
        with open(path.join(fpath, "trainer-metadata.data"), "rb") as f:
            trainermeta = pickle.load(f)
        return trainermeta



class Seq2SeqTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def train_step(self, batch: Batch) -> float:
        self.model.train()
        self.optim.zero_grad()
        outputs = self.model(input_ids = batch.input_ids, 
                                attention_mask = batch.attention_mask, 
                                decoder_input_ids = batch.labels[:, :-1], 
                                decoder_attention_mask = batch.decoder_attention_mask[:, :-1])
        next_tokens = batch.labels[:, 1:]
        # next_tokens_mask = batch.decoder_attention_mask[:, 1:]
        # next_tokens_replaced = torch.where(next_tokens_mask.bool(), next_tokens, -100)
        vocab_size = outputs.logits.shape[-1]
        loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), next_tokens.reshape(-1), ignore_index=self.tokenizer.pad_token_id)

        self.accelerator.backward(loss)
        self.optim.step()
        return loss.item()

    def val_step(self, batch: Batch) -> Tuple[float, bool]:
        self.model.eval() 

        outputs = self.model(input_ids = batch.input_ids, 
                                attention_mask = batch.attention_mask, 
                                decoder_input_ids = batch.labels[:, :-1], 
                                decoder_attention_mask = batch.decoder_attention_mask[:, :-1])
        next_tokens = batch.labels[:, 1:]
        vocab_size = outputs.logits.shape[-1]
        loss = F.cross_entropy(outputs.logits.view(-1, vocab_size), next_tokens.reshape(-1), ignore_index=self.tokenizer.pad_token_id)
        return loss.item(), False







if __name__ == "__main__":
    Env.setup("config/general.yaml")
    Env.info()

    model_hf = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
    model = BartTaskModel(tfmr=model_hf)
    tok = AutoTokenizer.from_pretrained("facebook/bart-large")

    optim = AdamW(model.parameters(), lr=1e-5)

    trainer = Seq2SeqTrainer(model, tok, DART, optim)
    trainer.run_loop()

