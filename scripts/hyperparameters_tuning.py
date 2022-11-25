from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from source.train import run_training
import torch
from source.evaluate import evaluate_on
from source.resources import EXP_DIR, Dataset
import optuna

def run_tuning(params):
    
    args_dict = dict(
        lang='en',
        model_name='t5-base',
        max_seq_length=78,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        warmup_steps=5,
        train_batch_size=8,
        valid_batch_size=8,
        num_train_epochs=params['num_train_epochs'],
        gradient_accumulation_steps=1, #16
        n_gpu=torch.cuda.device_count(),
        # early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='01', # 01, you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=12,
        nb_sanity_val_steps=0,
        topk=10
    )
    features_kwargs = {
        'CharRatioFeature': {'target_ratio': 0.8},
        'WordRankRatioFeature': {'target_ratio': 0.8},
    }
    args_dict['features_kwargs'] = features_kwargs
    run_training(args_dict, Dataset.LexMTurk)

    features_kwargs = {
        'CharRatioFeature': {'target_ratio': 0.7},
        'WordRankRatioFeature': {'target_ratio': 0.7},
    }
    return evaluate_on(Dataset.LexMTurk, features_kwargs, 'test', lang='en', verbose=False)


def objective(trial: optuna.trial.Trial) -> float:

    params = {
        'num_train_epochs': trial.suggest_int('num_train_epochs', 6, 20)
    }
    return run_tuning(params)

if __name__ == '__main__':
    tuning_log_dir = EXP_DIR 
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name='Hyperparameters', direction="maximize",
                                storage=f'sqlite:///{tuning_log_dir}/study.db', load_if_exists=True)
    study.optimize(objective, n_trials=20)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")