from dataclasses import dataclass
import typing
from typing import Optional, Tuple
import os 
from os import path 

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig

from k_dedede.env import Env



################################################
#              TASK DATACLASSES                #
################################################ 

@dataclass
class TransformerOutputs:
    logits: torch.Tensor = None 
    attentions: Tuple[torch.Tensor] = None
    hidden_states: Tuple[torch.Tensor] = None
class TaskArgs:
    def __init__(self, **kwargs):
        [setattr(self, k, kwargs[k]) for k in kwargs]



################################################
#                TASK FACTORY                  #
################################################ 

class TaskFactory:
    def task_map(task: str) -> TaskArgs:
        t_map = {
            "mnli": TaskArgs(task_type="classification", input_dim=768, n_classes=3)

        }
        try:
            return t_map[task]
        except:
            raise NotImplementedError(f"Task head not implemented for task {task}")
    def model_map(hf_id):
        m_map = {
            "roberta-base": RobertaTaskModel
        }
        try:
            return m_map[hf_id] 
        except:
            raise NotImplementedError(f"Does not support huggingface-id {hf_id}")
    def get_model_args_from_env():
        return Env.n_layers, Env.hidden_d, Env.model_d 
    def get_task_model(task: str, hf_id: str):
        tfmr = AutoModel.from_pretrained(hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id)

        TaskModelCls = TaskFactory.model_map(hf_id)
        task_args = TaskFactory.task_map(task)
        if task_args.task_type == "classification":
            task_head = ClassificationHead(task_args.input_dim, task_args.n_classes)
        elif task_args.task_type == "regression":
            task_head = RegressionHead(task_args.input_dim, task_args.output_dim)
        else:
            raise NotImplementedError() 

        n_layers, hidden_d, model_d = TaskFactory.get_model_args_from_env()
        # if any of these are defined, then model will start from random init
        if n_layers is not None or \
            hidden_d is not None or \
            model_d is not None:
            new_config = TaskModelCls.update_config(tfmr.config, n_layers, hidden_d, model_d)
            tfmr = AutoModel.from_config(new_config)
        task_model = TaskModelCls(tfmr=tfmr, head=task_head)
        return task_model
    




################################################
#                  TASK HEADS                  #
################################################ 



class TaskHead(nn.Module):
    def __init__(self, reduce_along_sequence:bool=True):
        super().__init__()
        self.reduce_along_sequence = reduce_along_sequence
    def forward(self): 
        raise NotImplementedError()
    def assert_reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce_along_sequence:
            assert x.dim() == 3, "input must have dimensions [N, S, C]"
            x = x.mean(dim=1)
        return x

class RegressionHead(TaskHead):
    def __init__(self, input_dim: int, output_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x: torch.Tensor):
        x = self.assert_reduction(x)
        x = self.linear(x)
        return x

class ClassificationHead(TaskHead): 
    def __init__(self, input_dim:int , n_classes: int, output_probs:bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_probs = output_probs
        self.linear = nn.Linear(input_dim, n_classes)
        if self.output_probs:
            if n_classes > 1:
                self.act = nn.Softmax(dim=-1)
            else:
                self.act = nn.Sigmoid()
        else:
            self.act = None
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.assert_reduction(x) 
        x = self.linear(x) 
        if self.act is not None:
            x = self.act(x) 
        return x


################################################
#                 TASK MODELS                  #
################################################ 

class TaskModel(nn.Module):
    def __init__(self, tfmr: PreTrainedModel, head: TaskHead):
        super().__init__()
        self.tfmr = tfmr
        self.head = head
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    def update_config(config: PretrainedConfig, 
                      n_layers:int, 
                      hidden_d:int, 
                      model_d:int) -> PretrainedConfig:
        raise NotImplementedError

class RobertaTaskModel(TaskModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        assert "roberta" in self.tfmr.config._name_or_path , "You must import a RoBERTa model"
    # TODO
    def forward(self, *args, **kwargs) -> TransformerOutputs:
        outputs = self.tfmr(*args, **kwargs)
        logits = self.head(outputs.last_hidden_state)
        tfmr_outputs = TransformerOutputs(
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
        )
        return tfmr_outputs
    def update_config(config: PretrainedConfig, 
                      n_layers: Optional[int], 
                      hidden_d: Optional[int], 
                      model_d: Optional[int]) -> PretrainedConfig:
        if n_layers is not None: 
            config.num_hidden_layers = n_layers 
        if hidden_d is not None:
            config.hidden_size = hidden_d 
        if model_d is not None:
            config.intermediate_size = model_d
        return config





if __name__ == "__main__": 
    TaskArgs(task_type="classification", input_dim=10, n_classes=5)
    print(vars(TaskArgs))
    tfmr_outputs = TransformerOutputs(1)
    print(vars(tfmr_outputs))

    taskmodel = TaskFactory.get_task_model("mnli", "roberta-base")
    x = torch.ones((8, 10), dtype=torch.long)
    attn_mask = x.clone()

    y = taskmodel(x, attention_mask=attn_mask)
    print(y)
