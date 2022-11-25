from pathlib import Path;import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))# -- fix path --

from source.evaluate import evaluate_on
from source.resources import Dataset


# evaluate using the last trained model
lang = 'en'
features_kwargs = {
        'CandidateRanking': {'target_ratio': 1.00},
        'WordLength': {'target_ratio': 0.70},
        'WordRank': {'target_ratio': 0.70},
    }

evaluate_on(Dataset.LexMTurk, features_kwargs, 'test', lang=lang)
evaluate_on(Dataset.NNSeval, features_kwargs, 'test', lang=lang)
evaluate_on(Dataset.BenchLS, features_kwargs, 'test', lang=lang)

