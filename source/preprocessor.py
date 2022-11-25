# # -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# # -- end fix path --

from source.metrics import flatten, sort_candidates_by_frequency, sort_candidates_by_ranking

from source.helper import tokenize, unique, yield_lines, load_dump, dump, write_lines, count_line, \
    print_execution_time, save_preprocessor
from source.resources import DUMPS_DIR, Dataset, RESOURCES_DIR, download_fasttext_embedding, PROCESSED_DATA_DIR, get_dataset_filepath
from nltk.corpus import stopwords
from source import helper, wandb_config
from functools import lru_cache
from string import punctuation
import numpy as np
import spacy
import nltk
import pandas as pd
from wordfreq import word_frequency
from hyphen import Hyphenator
import re
import json 
    
nltk.download('stopwords')


def round(val):
    return f'{val:.2f}'


def safe_division(a, b):
    return a / b if b else 0

@lru_cache(maxsize=5)
def get_stopwords(lang):
    if lang == 'es':
        return set(stopwords.words('spanish'))
    elif lang == 'fr':
        return set(stopwords.words('french'))
    elif lang == 'en':
        return set(stopwords.words('english'))
    elif lang == 'pt':
        return set(stopwords.words('portuguese')) 
    elif lang == 'de':
        return set(stopwords.words('german'))
    else:
        return None
    
    
@lru_cache(maxsize=1024)
def is_punctuation(word):
    return not ''.join([char for char in word if char not in punctuation])


@lru_cache(maxsize=128)
def remove_punctuation(text):
    return ' '.join([word for word in tokenize(text) if not is_punctuation(word)])


def remove_stopwords(text, lang):
    stopwords = get_stopwords(lang)
    return ' '.join([w for w in tokenize(text) if w.lower() not in stopwords])

def word_index_in_text(text, word):
    # words = text.lower().split(' ')
    words = tokenize(text.lower())
    return words.index(word.lower()) 

@lru_cache(maxsize=1)
def get_word2rank(lang, vocab_size=np.inf):
    filename = f'cc.{lang}.300.bin'
    model_filepath = DUMPS_DIR / f"{filename}.pk"
    if model_filepath.exists():
        return load_dump(model_filepath)
    print("Preprocessing word2rank...")
    word_embeddings_filepath = download_fasttext_embedding(lang)
    lines_generator = yield_lines(word_embeddings_filepath)
    word2rank = {}
    # next(lines_generator)
    for i, line in enumerate(lines_generator):
        if i >= vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    dump(word2rank, model_filepath)
    return word2rank


def download_requirements(lang):
    get_word2rank(lang)

def encode_token(text, source_word, target_word):
    # return text.replace(source_word, f'[T]{target_word}[/T]')
    pattern = re.compile(source_word, re.IGNORECASE)
    return pattern.sub(f'[T]{target_word}[/T]', text)

def __normalize_string(text):
        text = text.replace('``', '"')
        text = text.replace('`', "'")
        text = text.replace("''", '"')
        text = text.strip('"')

        return text 
    
def __load_data_with_index(dataset):
        dataset_filepath = get_dataset_filepath(dataset)
        lines = yield_lines(dataset_filepath)
        docs = []

        for line in lines:
            line = line.strip().lower()
            chunks = line.split('\t')
            text = chunks[0].strip()
            text = __normalize_string(text)
            complex_word = chunks[1].strip()
            candidates = chunks[3:]
            candidates = [tuple(candidate.split(':')) for candidate in candidates]
            candidates = [(word.strip(), index) for index, word in candidates]
            
            candidates = sort_candidates_by_ranking(candidates)
            
            doc = {'text': text,
                        'complex_word': complex_word,
                        'complex_word_index': chunks[2],
                        'candidates': json.dumps(candidates)}
            docs.append(doc)
        return pd.DataFrame(docs)
    

def __load_data(dataset):
    dataset_filepath = get_dataset_filepath(dataset)
    lines = yield_lines(dataset_filepath)
    docs = []

    for line in lines:
        line = line.strip().lower()
        chunks = line.split('\t')
        text = chunks[0].strip()
        text = __normalize_string(text)
        complex_word = chunks[1].strip()
        candidates = chunks[2:]
        candidates = [word.strip() for word in candidates] 
        candidates = sort_candidates_by_frequency(candidates)
        doc = {'text': text,
                    'complex_word': complex_word,
                    'complex_word_index': word_index_in_text(text, complex_word),
                    'candidates': json.dumps(candidates)}
        docs.append(doc)
    return pd.DataFrame(docs)

def load_data(dataset):
    lines = yield_lines(get_dataset_filepath(dataset))
    next(lines)
    line = next(lines)
    chunks = line.split('\t')
    if chunks[2].isdigit():
        data = __load_data_with_index(dataset)
    else:
        data = __load_data(dataset)
        
    return data 
class RatioFeature:
    def __init__(self, class_name, feature_extractor, target_ratio=0.8, lang='en'):
        self.lang = lang
        self.class_name = class_name
        self.feature_extractor = feature_extractor
        self.target_ratio = f'{target_ratio:.2f}'
        
    def get_target_ratio(self):
        return f'{self.name}_{self.target_ratio}'
   
    def extract_ratio(self, complex_word, simple_word, sorted_candidates):
        return f'{self.name}_{self.feature_extractor(complex_word, simple_word, sorted_candidates)}'

    @property
    def name(self):
        # class_name = self.__class__.__name__.replace('RatioFeature', '')
        # return ''.join(word[0] for word in re.findall('[A-Z][^A-Z]*', class_name) if word) or class_name
        return self.class_name or self.__class__.__name__.replace('RatioFeature', '')


class WordLength(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WL', self.get_char_length_ratio, *args, **kwargs)

    def get_char_length_ratio(self, complex_word, simple_word, *args):
        return round(safe_division(len(simple_word), len(complex_word)))


class WordRank(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WR', self.get_word_rank_ratio, *args, **kwargs)

    def get_word_rank_ratio(self, complex_word, simple_word, *args):
        return round(safe_division(self.get_lexical_complexity_score(simple_word),
                                       self.get_lexical_complexity_score(complex_word)))

    def get_lexical_complexity_score(self, word):
        # we tokenize it it because it could be a phrase
        words = tokenize(word)
        words = [word for word in words if word in get_word2rank(self.lang)]
        if not words:
            return np.log(1 + len(get_word2rank(self.lang)))
        # return np.quantile([self.get_normalized_rank(word) for word in words], 0.75)
        return np.mean([self.get_normalized_rank(word) for word in words])

    @lru_cache(maxsize=10000)
    def get_normalized_rank(self, word):
        max = len(get_word2rank(self.lang))
        rank = get_word2rank(self.lang).get(word, max)
        return np.log(1 + rank) / np.log(1 + max)
        # return np.log(1 + rank)

class WordSyllable(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('WS', self.get_word_syllable_ratio, *args, **kwargs)

    @lru_cache(maxsize=1)
    def get_hypernator(self):
        if self.lang == 'en':
            return Hyphenator('en_US')
        elif self.lang == 'es':
            return Hyphenator('es')
        elif self.lang == 'fr':
            return Hyphenator('fr')
        elif self.lang == 'pt':
            return Hyphenator('pt')
        else: 
            return None

    @lru_cache(maxsize=10**6)
    def count_syllable(self, word):
        h = self.get_hypernator()  
        return len(h.syllables(word))

    def get_word_syllable_ratio(self, complex_word, simple_word, *args):
        return round(safe_division(self.count_syllable(simple_word), self.count_syllable(complex_word)))

class CandidateRanking(RatioFeature):
    def __init__(self, *args, **kwargs):
        super().__init__('CR', self.get_ranking_ratio, *args, **kwargs)
    
    def get_ranking_ratio(self, complex_word, simple_word, grouped_candidates):
        ranks = {0: 1.00, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.00}    
        index = 4
        for i, candidates in enumerate(grouped_candidates):
            if simple_word in candidates:
                index = i
                break
        if index > 4: index = 4
        return round(ranks[index])
        
class Preprocessor:
    def __init__(self, features_kwargs, lang):
        super().__init__()
        self.lang = lang 
        
        self.features = self.__get_features(features_kwargs)
        if features_kwargs:
            self.hash = helper.generate_hash(str(features_kwargs).encode())
            self.num_feature = len(features_kwargs)
        else:
            self.hash = "no_feature"
            self.num_feature = 0

    def __get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs, lang=self.lang)

    def __get_features(self, feature_kwargs):
        return [self.__get_class(feature_name, **kwargs) for feature_name, kwargs in feature_kwargs.items()]
    
    def get_hash(self):
        return self.hash
    
    def decode_sentence(self, encoded_sentence):
        for feature in self.features:
            decoded_sentence = feature.decode_sentence(encoded_sentence)
        return decoded_sentence

    def extract_ratios(self, text, complex_word, simple_word, candidates):
        if not self.features:
            return ''
        ratios = ''
        for feature in self.features:
            val = feature.extract_ratio(complex_word, simple_word, candidates)
            ratios += f'<{val}> '
        return ratios.strip()
    
    
    def encode_sentence(self, text, complex_word):
        text = encode_token(text, complex_word, complex_word)
        return f'simplify: {self.get_target_ratios()} {text}'
    
    
    def get_target_ratios(self):
        if not self.features:
            return ''
        ratios = ''.join(f'<{feature.get_target_ratio()}> ' for feature in self.features)
        return ratios.rstrip()
    
        
    def __split_data(self, data, frac=0.8, seed=42):
        data_train = data.sample(frac=frac, random_state=seed)
        data_valid = data.drop(data_train.index)
        return data_train, data_valid     
        
    def __split_data_train_valid_test(self, data, frac=0.8, seed=42):
        data_train, data_valid = self.__split_data(data, frac, seed)
        data_valid, data_test = self.__split_data(data_valid, frac=0.5, seed=seed)
        return data_train, data_valid, data_test
    
    def preprocess(self, dataset, seed=42):
        download_requirements(self.lang)
        save_preprocessor(self)
        self.output_dir = PROCESSED_DATA_DIR / dataset 
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        data_filepath = self.output_dir / f'{dataset}.csv'
        train_filepath = self.output_dir / f'{dataset}.train.csv'
        valid_filepath = self.output_dir / f'{dataset}.valid.csv'
        train_processed_filepath = self.output_dir / f'{dataset}.train.{self.hash}.csv'
        if not all([train_filepath.exists(), valid_filepath.exists(), train_processed_filepath.exists()]):
            data = load_data(dataset)
                
            train_set, valid_set = self.__split_data(data, frac=0.9, seed=seed)
            data.to_csv(data_filepath, index=False)
            train_set.to_csv(train_filepath, index=False)
            valid_set.to_csv(valid_filepath, index=False)
            train_set['candidates'] = train_set['candidates'].apply(lambda x: json.loads(x))
            
            processed_doc= []
            for i in range(len(train_set)):
                row = train_set.iloc[i]
                text = row['text']
                complex_word = row['complex_word']
                grouped_candidates = row['candidates']
                candidates = flatten(grouped_candidates)
                for simple_word in candidates:
                    ratios = self.extract_ratios(text, complex_word, simple_word, grouped_candidates)
                    
                    complex_sent = encode_token(text, complex_word, complex_word)
                    simple_sent = encode_token(text, complex_word, simple_word)
                    item = {'complex': f'{ratios} {complex_sent}',
                            'simple': simple_sent}
                    processed_doc.append(item)

            pd.DataFrame(processed_doc).to_csv(train_processed_filepath, index=False)
            
        return self.output_dir
    

    def load_preprocessed_data(self, dataset, phase):
        data_dir = PROCESSED_DATA_DIR / dataset
        if phase == 'train':
            filepath = data_dir / f'{dataset}.{phase}.{self.hash}.csv'
        else:
            filepath = data_dir / f'{dataset}.{phase}.csv'
            
        return pd.read_csv(filepath)

