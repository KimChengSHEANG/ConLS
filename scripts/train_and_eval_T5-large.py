from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))# -- fix path --

from source.evaluate import evaluate_on
from source.resources import Dataset
from source.train import train_on
import torch
lang = 'en'
args_dict = dict(
    lang=lang,
    model_name='t5-large',
    max_seq_length=128, #max 78
    learning_rate=1e-5, #3e-4
    weight_decay=0.1,
    adam_epsilon=1e-8,
    warmup_steps=5,
    train_batch_size=8,
    valid_batch_size=8,
    num_train_epochs=8,
    gradient_accumulation_steps=1, #16
    n_gpu=torch.cuda.device_count(),
    # early_stop_callback=False,
    fp_16=False, 
    opt_level='01', 
    max_grad_norm=1.0, 
    seed=12,
    nb_sanity_val_steps=1,
)


features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 0.80},
        'WordRank': {'target_ratio': 0.80},
    }
args_dict['features_kwargs'] = features_kwargs

features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 0.70},
        'WordRank': {'target_ratio': 0.70},
    }

train_on(Dataset.TSAR_EN, args_dict)

evaluate_on(Dataset.LexMTurk, features_kwargs, 'test', lang=lang)
evaluate_on(Dataset.NNSeval, features_kwargs, 'test', lang=lang)
evaluate_on(Dataset.BenchLS, features_kwargs, 'test', lang=lang)