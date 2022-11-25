import sys; from pathlib import Path; sys.path.append(str(Path(__file__).resolve().parent.parent)) # fix source path

from source.preprocessor import Preprocessor
from source.resources import Dataset, Language 

if __name__ == '__main__':
    features_kwargs = {
        'RankingRatioFeature': {'target_ratio': 1.00},
        'CharRatioFeature': {'target_ratio': 0.70},
        'WordRankRatioFeature': {'target_ratio': 0.70},
        'WordSyllableRatioFeature': {'target_ratio': 0.70},
    }
    preprocessor = Preprocessor(features_kwargs, lang=Language.english)
    preprocessor.preprocess(Dataset.LexMTurk)