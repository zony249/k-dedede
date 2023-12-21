from typing import List
from yaml import safe_load
from argparse import ArgumentParser

class Env:
    """
    Env provides default arguments for basic training requirements.
    Arguments are read from the config file. New arguments can be specified.
    """
    # General env args 
    data = "./data/"
    dataset_cache = "./data/cache/"
    rebuild_dataset = False
    output_dir = "runs"
    exp_name = None
    device = "cpu"

    # task args
    task = None 
    tfmr = None 

    # training args
    lr = 1e-5    
    batch_size = 16
    epochs = 1
    optimizer = "adamw"
    scheduler = None
    eval_every_steps = 500

    # model config
    n_layers = None 
    hidden_d = None 
    model_d = None 

    # methods
    def info():
        print("Environment:")
        [print(f"\t{v}={m}") for v, m in vars(Env).items() if not (v.startswith('_')  or callable(m))]
    def set_attributes(**kwargs):
        [setattr(Env, k, kwargs[k]) for k in kwargs if k in vars(Env)]
        if len([k for k in kwargs if k not in vars(Env)]):
            print("The following attributes were not set:")
            [print(f"\t{k}") for k in kwargs if k not in vars(Env)]
    def from_yaml(yaml_file=str):
        with open(yaml_file, "r") as f:
            config = safe_load(f)
            Env.set_attributes(**config)
    def setup(config:str):
        Env.from_yaml(config)

if __name__ == "__main__":
    Env.info()
    Env.set_attributes(tfmr="roberta-base", task = "wiki", burger = "king")
    Env.info()
    Env.from_yaml("config/general.yaml")
    Env.info()