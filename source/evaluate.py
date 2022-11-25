import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from source.metrics import MAP_at_k, accuracy_at_1, accuracy_at_k_at_top_gold_1, flatten, normalize, precision_metrics_at_k, remove_word_from_list
from source.generate import generate
from pytorch_lightning import seed_everything
from source.model import FineTunerModel
import pandas as pd
import json
import gc
from source.preprocessor import Preprocessor, load_data
import torch
from source.resources import get_dataset_filepath, get_last_experiment_dir, EXP_DIR
from source.helper import count_line, log_stdout, read_lines, unique, write_lines
from source import wandb_config
import re, os
import wandb 
import json 
import math 

seed_everything(12)

def load_model(model_dirname=None):

    if model_dirname is None:  # default
        model_dir = get_last_experiment_dir()
    else:
        model_dir = EXP_DIR / model_dirname

    print("Load model", model_dirname)
    print("Model dir: ", model_dir)
    params_filepath = model_dir / "params.json"
    params = json.load(params_filepath.open('r'))
    params['model_dir'] = model_dir

    
    checkpoints = list(model_dir.glob('checkpoint*'))
    best_checkpoint = sorted(checkpoints, reverse=True)[0]
    
    print('check_point:', best_checkpoint)
    print("loading model...")
    
    checkpoint = FineTunerModel.load_from_checkpoint(checkpoint_path=best_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = checkpoint.model.to(device)
    return model, checkpoint.tokenizer, params

    
def evaluate_on(dataset, features_kwargs, phase, lang, model_dirname=None):
    model, tokenizer, params = load_model(model_dirname)
    
    params['eval_features'] = features_kwargs
    params['lang'] = lang 
    
    max_len = int(params['max_seq_length'])

    os.environ['WANDB_API_KEY'] = wandb_config.WANDB_API_KEY
    os.environ['WANDB_MODE'] = wandb_config.WANDB_MODE
    wandb.init(project=wandb_config.WANDB_PROJECT_NAME, name=f"{params['model_dir'].stem}_Eval", job_type='Evaluate', config=params)
    

    preprocessor = Preprocessor(features_kwargs, lang)
    output_dir = params['model_dir'] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output dir: ", output_dir)
    # features_hash = preprocessor.get_hash()
    

    # if not output_score_filepath.exists() or count_line(output_score_filepath) == 0:
    data = load_data(dataset)
    # log_params(output_dir / f"{features_hash}_features_kwargs.json", features_kwargs)
    # start_time = time.time()
    values = [f'{features_kwargs[key]["target_ratio"]:.2f}' for key in features_kwargs.keys()]    
    values = '_'.join(values)

    num_file = 0
    output_score_filepath = output_dir / f"{dataset}.{phase}.{values}.log.{num_file}.txt" 
    while output_score_filepath.exists():
        num_file += 1
        output_score_filepath = output_dir / f"{dataset}.{phase}.{values}.log.{num_file}.txt" 
    
    pred_filepath = output_dir / f'{dataset}_{values}.tsv'
    pred_scores_filepath = output_dir / f'scores_{dataset}_{values}.csv'
    print(pred_filepath)
    if pred_filepath.exists() and count_line(pred_filepath) >= len(data):
        print("File is already processed.")
    else:
        with log_stdout(output_score_filepath):
            complex_words = []
            list_pred_candidates = []
            list_gold_candidates = []
            list_sorted_gold_candidates = []
            
            pred_sents_list = []
            output = []
            logs = []
            for i in range(len(data)):
                row = data.iloc[i]
                complex_word = row["complex_word"]
                complex_words.append(complex_word)
                gold_candidates = json.loads(row['candidates'])
                list_gold_candidates.append(unique(flatten(gold_candidates)))
                list_sorted_gold_candidates.append(gold_candidates)
                
                source = preprocessor.encode_sentence(row['text'], row['complex_word'])
                pred_sents, pred_candidates = generate(source, model, tokenizer, max_len)
                pred_sents_list.append(pred_sents)
                
                # if verbose:
                print(f'{i}/{len(data)}', '='*80)
                print(source)
                # print(f'Unique candidates: {unique(gold_candidates)}')
                # print('\n'.join(pred_sents))
                
                pred_candidates = remove_word_from_list(complex_word, pred_candidates) # remove candidates the same as complex word
                pred_candidates = pred_candidates[:10] # limit it to max of 10
                print(f'Gold candidates: ', gold_candidates)
                print(f'Complex word: {complex_word}')
                print(f'Predicted candidates: ', pred_candidates)
                list_pred_candidates.append(pred_candidates)
                output.append(f'{row["text"]}\t{row["complex_word"]}\t' + '\t'.join(pred_candidates))
            
            write_lines(output, pred_filepath)
            
            
            print("Features: ")  
            for key, val in features_kwargs.items():
                print(f'{key:<15}', ':', val['target_ratio'])              
                            
            to_save_data = {}
            
            # Acc@1
            value = accuracy_at_1(list_pred_candidates, list_gold_candidates)
            value = normalize(value)
            log_label = f'ACC@1'
            print(f'{log_label:>30}: {value}')
            logs.append(f'{log_label:>30}:\t{value}')
            wandb.log({f'AC@1': value})
            to_save_data[log_label] = value
            
            
            # compute accuracy@1@Top1
            for k in [1, 2, 3]:
                value = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_sorted_gold_candidates, k)
                value = normalize(value)    
                print('='*20, f' Accuracy at k:{k} at top gold 1 ', '='*20)
                log_label = f'ACC@{k}@Top1'
                print(f'{log_label:>30}: {value}')
                logs.append(f'{log_label:>30}:\t{value}')
                wandb.log({f'ACC@{k}@Top1': value})
                to_save_data[log_label] = value
                
            # compute MAP
            for k in [1, 3, 5]:
                value = MAP_at_k(list_pred_candidates, list_gold_candidates, k)
                value = normalize(value)
                print('='*20, f' MAP at k:{k} at top gold 1 ', '='*20)
                log_label = f'MAP@{k}'
                print(f'{log_label:>30}: {value}')
                logs.append(f'{log_label:>30}:\t{value}')
                wandb.log({f'MAP@{k}': value})
                to_save_data[log_label] = value
            
            for k in [1, 3, 5]:
                print('='*20, f' Precision metrics at k:{k} ', '='*20)
                scores = precision_metrics_at_k(list_pred_candidates, list_gold_candidates, k)
                print(f'\n@{k}')
                for key, value in scores.items():
                    value = normalize(value)
                    log_label = f'{key}@{k}'
                    print(f'{log_label:25}:\t{value}') 
                    logs.append(f'{log_label:>30}:\t{value}')
                    wandb.log({f'{key}@{k}': f'{value}'})
                    to_save_data[log_label] = value
            
            
            pd.DataFrame([to_save_data]).to_csv(pred_scores_filepath, index=False)
                
            print('='*80) 
            logs = sorted(logs)
            for log in logs:
                print(log)
            
            # print("Execution time: --- %s seconds ---" % (time.time() - start_time))
    # else:
    #     print("Already exist: ", output_score_filepath)
    #     print("".join(read_lines(output_score_filepath)))
    del model
    del tokenizer
    gc.collect() # clean up memory
    wandb.finish()


