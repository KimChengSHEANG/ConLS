from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


from source.evaluate import evaluate_on
from source.resources import EXP_DIR, Dataset
import optuna

def run_tuning(params):
    feature_kwargs = {
            'CharRatioFeature': {'target_ratio': params['CharRatioFeature']},
            'WordRankRatioFeature': {'target_ratio': params['WordRankRatioFeature']},
        }
    print(feature_kwargs)
    return evaluate_on(Dataset.LexMTurk, feature_kwargs, 'test')


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'CharRatioFeature': trial.suggest_float('CharRatioFeature', 0.6, 1.0, step=0.05),
        'WordRankRatioFeature': trial.suggest_float('WordRankRatioFeature', 0.6, 1.0, step=0.05),
    }
    return run_tuning(params)

if __name__ == '__main__':
    tuning_log_dir = EXP_DIR 
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name='Tokens', direction="maximize",
                                storage=f'sqlite:///{tuning_log_dir}/tokens_study.db', load_if_exists=True)
    study.optimize(objective, n_trials=500)

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")