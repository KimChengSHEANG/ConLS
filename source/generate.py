from timeit import repeat
import torch
import re
from source.helper import unique 

def extract_substitute(text):
    matches = re.findall(r'\[T\](.*?)\[\/T\]', text) # (.*?) match the first occurance 
    return matches[0] if matches else ''


def generate(source, model, tokenizer, max_seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = tokenizer(
        source,
        truncation=True,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_masks = encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=False,
        max_length=max_seq_length,
        num_beams=15,
        top_k=120,
        top_p=0.98,
        repetition_penalty=2.0,
        temperature=0.9,
        early_stopping=True,
        num_return_sequences=15
    )
    pred_sents = []
    pred_candidates = []
    for output in beam_outputs:
        sent = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        substitute = extract_substitute(sent).strip()
        pred_sents.append(f'[{substitute}]\t{sent}')
        pred_candidates.append(substitute)
        
    pred_candidates = unique([item for item in pred_candidates]) # remove empty word and make it unique
    return pred_sents, pred_candidates
    