# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # fix path

import math
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import magic

def get_file_encoding(filepath):
    blob = open(filepath, 'rb').read()
    m = magic.Magic(mime_encoding=True)
    return m.from_buffer(blob)


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as fout:
        for line in lines:
            fout.write(line + '\n')
            
def yield_lines(filepath):
    filepath = Path(filepath)
    with filepath.open('r', encoding=get_file_encoding(filepath)) as f:
        for line in f:
            yield line.strip() 


def safe_division(a, b):
    return a / b if b else 0
           
def unique(items): 
    # filter duplicate and preserve order
    return list(dict.fromkeys(items))

def remove_word_from_list(word, word_list):
    return [w for w in word_list if w.lower() != word.lower()]

def sort_candidates_by_ranking(candidates):
    # it is for NNSeval, BenchLS, ... 
    # candidates: [('parts', 1), ('component', 2), ('sections', 2), ...]
    
    dict_candidates = defaultdict(list)
    for word, rank in candidates:
        dict_candidates[rank].append(word)
    
    sorted_keys = sorted(dict_candidates.keys())
    return [dict_candidates[key] for key in sorted_keys]
    

def sort_candidates_by_frequency(candidates):
    ''' lex.mturk, tsar, 
    Return ranked candidates in groups: [['parts'], ['bits'], ['components'], ['component', 'sections', 'elements', 'part', 'information', 'items']]
    '''
    candidates = Counter(candidates).items()
    # [('parts', 40), ('component', 1), ('sections', 1), ('elements', 1), ('part', 1), ('components', 2), ('bits', 3), ('information', 1), ('items', 1)
    dict_candidates = defaultdict(list)
    for word, freq in candidates:
        dict_candidates[freq].append(word)
        
    sorted_keys = sorted(dict_candidates.keys(), reverse=True)
    return [dict_candidates[key] for key in sorted_keys]

def flatten(grouped_candidates):
    return [item for items in grouped_candidates for item in items]
    
def normalize(value):
    return round(value, 4)

def match(pred_candidate, gold_candidates):
    gold_candidates = [candidate.lower() for candidate in gold_candidates]
    pred_candidate = pred_candidate.lower()
    return pred_candidate in gold_candidates

def match_group(pred_candidates, gold_candidates):
    return any([match(pred, gold_candidates) for pred in pred_candidates])
    
def accuracy_at_1(list_pred_candidates, list_gold_candidates):
    # Accuracy Metric
    tp = 0
    total = 0
    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        if pred_candidates and len(pred_candidates[0]) > 0:
            if match(pred_candidates[0], gold_candidates):
                tp += 1
        total += 1

    accuracy = safe_division(tp, total)
    return normalize(accuracy)

def accuracy_at_k_at_top_gold_1(list_pred_candidates, list_sorted_gold_candidates, k):
    # Accuracy Metric
    tp = 0
    total = 0
    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_sorted_gold_candidates):
        if pred_candidates and len(pred_candidates[:k]) > 0:
            if match_group(pred_candidates[0:k], gold_candidates[0]):
                tp += 1
        total += 1

    accuracy = safe_division(tp, total)
    return normalize(accuracy)

def precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k):

    # Precision
    precision = 0
    recall = 0
    f1 = 0

    running_precision = 0
    running_recall = 0

    potential_counts = 0
    potential = 0

    total = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        labels = pred_candidates[:k]
        if len(labels) > 0:
            acc_labels = [l for l in labels if match(l, gold_candidates)]
            acc_gold = [l for l in gold_candidates if match(l, labels)]

            if len(acc_labels) > 0:
                potential_counts += 1

            precision = safe_division(len(acc_labels), len(labels))
            recall = safe_division(len(acc_gold), len(gold_candidates))

            running_precision += precision
            running_recall += recall
        
        total += 1

    precision = safe_division(running_precision, total)
    recall = safe_division(running_recall, total)


    f1 = 0
    if (precision + recall) > 0:
        f1 = safe_division(2 * precision * recall, (precision + recall))

    if (potential_counts > 0):
        potential = safe_division(potential_counts, total)

    # return normalize(precision), normalize(recall), normalize(f1), normalize(potential)
    return {'precision': precision, 
            'recall': recall, 
            'f1': f1,
            'potential': potential
            }

# Mean Average Precision
# Parameters :
#  1. List of Binary Relevance Judgments e.g. [False, True, True, False, False]
#  2. K

def compute_local_MAP(list_gold_items_match, k):
    list_gold_items_match = list_gold_items_match[:k]
    AP = 0
    TruePositivesSeen = 0
    for index, item in enumerate(list_gold_items_match, start=1):
        if item == True:
            TruePositivesSeen += 1
            precision = safe_division(TruePositivesSeen, index)
            AP += precision

    return safe_division(AP, k)

def MAP_at_k(list_pred_candidates, list_gold_candidates, k):

    total_instances = 0
    MAP_global_accumulator = 0

    for pred_candidates, gold_candidates in zip(list_pred_candidates, list_gold_candidates):
        
        labels_relevance_judgements = [match(label, gold_candidates) for label in pred_candidates]
        MAP_local = compute_local_MAP(labels_relevance_judgements, k)
        MAP_global_accumulator += MAP_local
        total_instances += 1

    MAP = 0
    if (MAP_global_accumulator > 0):
        MAP = safe_division(MAP_global_accumulator, total_instances)
    return MAP


class Evaluator(object):

    def __init__(self, filter_complex_word=True):
        self.filter_complex_word = filter_complex_word
    
    def read_files(self, pred_filepath, gold_filepath):
        self.sentences = []
        self.complex_words = []
        self.list_gold_candidates = []
        self.list_sorted_gold_candidates = []
        
        # load gold candidates
        for line in yield_lines(gold_filepath):
            # line = line.lower()
            
            chunks = line.split('\t')
            self.sentences.append(chunks[0])
            complex_word = chunks[1]
            self.complex_words.append(complex_word)
            candidates = chunks[2:]
            if self.filter_complex_word:
                candidates = remove_word_from_list(complex_word, candidates) 
                
            self.list_gold_candidates.append(unique(candidates))
            sorted_candidates = sort_candidates_by_frequency(candidates)
            self.list_sorted_gold_candidates.append(sorted_candidates)
        
        # load predicted candidates
        self.list_pred_candidates = []
        for line in yield_lines(pred_filepath):
            # line = line.lower()
            chunks = line.split('\t')
            complex_word = chunks[1]
            candidates = chunks[2:]
            if self.filter_complex_word:
                candidates = remove_word_from_list(complex_word, candidates) 
            candidates = unique(candidates)
            self.list_pred_candidates.append(candidates)
            
    def computeAccuracy_at_1(self):
        return accuracy_at_1(self.list_pred_candidates, self.list_gold_candidates)

    def computeAccuracy_at_N_at_top_gold_1(self, N):
        return accuracy_at_k_at_top_gold_1(self.list_pred_candidates, self.list_sorted_gold_candidates, N)

    def computePrecisionMetrics_at_K(self, k):
        return precision_metrics_at_k(self.list_pred_candidates, self.list_gold_candidates, k)
    
    def computeMAP_at_K(self, k):
        return MAP_at_k(self.list_pred_candidates, self.list_gold_candidates, k)
    
def evaluate_file(pred_filepath, gold_filepath, results_filepath):
    evaluator = Evaluator()
    evaluator.read_files(pred_filepath, gold_filepath)
    results = {}
    results[f'ACC@1'] = normalize(evaluator.computeAccuracy_at_1())
    # compute accuracy
    for k in [1, 2, 3, 4, 5]:
        results[f'ACC@{k}@Top1'] = normalize(evaluator.computeAccuracy_at_N_at_top_gold_1(k))
        
    # compute MAP
    for k in [1, 2, 3, 4, 5, 10]:
        results[f'MAP@{k}'] = normalize(evaluator.computeMAP_at_K(k))
    
    # compute potential, precision, Reecall, ... 
    tmp_result = {'potential': [], 'precision': [], 'recall': []} 
    for k in [1, 2, 3, 4, 5, 10]:
        
        values = evaluator.computePrecisionMetrics_at_K(k)
        tmp_result['potential'].append((f'Potential@{k}', values['potential']))
        tmp_result['precision'].append((f'Precision@{k}', values['precision']))
        tmp_result['recall'].append((f'Recall@{k}', values['recall']))
                                    
    for key in tmp_result:
        for metric, value in tmp_result[key]:
            results[metric] = value
            
    # for (key, value) in results.items():
    #     print(f'{key:<15}: {value}')
        
    pd.DataFrame([results]).to_csv(results_filepath, index=False)
    
if __name__=='__main__':
    evaluator = Evaluator()
    gold_filepath =  Path('resources') / 'datasets/lex.mturk.txt'
    results_dir = Path('results')
    pred_filepath = results_dir / 'lex.mturk.txt_outputs.txt'
    results_filepath = results_dir / f'{pred_filepath.stem}.csv'
    evaluator.read_files(pred_filepath, gold_filepath)
    results = {}
    results[f'ACC@1'] = normalize(evaluator.computeAccuracy_at_1())
    # compute accuracy
    for k in [1, 2, 3, 4, 5]:
        results[f'ACC@{k}@Top1'] = normalize(evaluator.computeAccuracy_at_N_at_top_gold_1(k))
        
    # compute MAP
    for k in [1, 2, 3, 4, 5, 10]:
        results[f'MAP@{k}'] = normalize(evaluator.computeMAP_at_K(k))
    
    # compute potential, precision, Reecall, ... 
    tmp_result = {'potential': [], 'precision': [], 'recall': []} 
    for k in [1, 2, 3, 4, 5, 10]:
        
        values = evaluator.computePrecisionMetrics_at_K(k)
        tmp_result['potential'].append((f'Potential@{k}', values['potential']))
        tmp_result['precision'].append((f'Precision@{k}', values['precision']))
        tmp_result['recall'].append((f'Recall@{k}', values['recall']))
                                    
    for key in tmp_result:
        for metric, value in tmp_result[key]:
            results[metric] = value
            

    for (key, value) in results.items():
        print(f'{key:<15}: {value}')
        
    pd.DataFrame([results]).to_csv(results_filepath, index=False)