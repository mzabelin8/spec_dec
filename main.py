import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_dec import speculative_generate, normal_generate
from experiments_func import run_experiment, create_plots

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_models(big_model_name, small_model_name):
    model_3b = AutoModelForCausalLM.from_pretrained(big_model_name).to(device)
    model_1b = AutoModelForCausalLM.from_pretrained(small_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(big_model_name)
    return model_3b, model_1b, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run speculative decoding experiments')
    parser.add_argument('--big_model', type=str, default="unsloth/Llama-3.2-3B-Instruct",
                        help='Name or path of the big model (default: unsloth/Llama-3.2-3B-Instruct)')
    parser.add_argument('--small_model', type=str, default="unsloth/Llama-3.2-1B-Instruct",
                        help='Name or path of the small model (default: unsloth/Llama-3.2-1B-Instruct)')
    parser.add_argument('--output', type=str, default="experiment_results.csv",
                        help='Output CSV file for experiment results (default: experiment_results.csv)')
    
    args = parser.parse_args()
    
    model_3b, model_1b, tokenizer = load_models(args.big_model, args.small_model)
    
    results_df = run_experiment(model_3b, model_1b, tokenizer)
    create_plots(results_df)
    results_df.to_csv(args.output, index=False)

