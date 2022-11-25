
import pytorch_lightning as pl
from torch.nn import functional as F
import wandb
from source.generate import generate
from source.helper import load_preprocessor
import nltk
from torch.utils.data import Dataset, DataLoader
from source.metrics import accuracy_at_1, accuracy_at_k_at_top_gold_1, remove_word_from_list
from source.resources import Dataset

import json 
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,get_linear_schedule_with_warmup, AutoConfig,
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
)

nltk.download('punkt')
class FineTunerModel(pl.LightningModule):
    def __init__(self, lang, model_name, learning_rate, adam_epsilon, weight_decay, dataset,
                 train_batch_size, valid_batch_size, max_seq_length,
                 n_gpu, gradient_accumulation_steps, num_train_epochs, warmup_steps, nb_sanity_val_steps,
                 *args, **kwargs):
        super(FineTunerModel, self).__init__()
        self.save_hyperparameters()

        # if 'bart' in self.hparams.model_name:
        #     self.model = BartForConditionalGeneration.from_pretrained(self.hparams.model_name)
        #     self.tokenizer = BartTokenizer.from_pretrained(self.hparams.model_name)
            
        #     # self.model = MBartForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        #     # self.tokenizer = MBart50Tokenizer.from_pretrained(self.hparams.model_name_or_path, src_lang="es_XX", tgt_lang="es_XX")
        
        # elif 't5' in self.hparams.model_name: 
        #     self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        #     self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name)
            
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        
        self.model = self.model.to(self.device)
        self.preprocessor = load_preprocessor()

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels)

    
    def training_step(self, batch, batch_idx):
        
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 
        
        outputs = self(input_ids=batch["source_ids"], 
                       attention_mask=batch["source_mask"], 
                       labels=labels, 
                       decoder_attention_mask=batch['target_mask'])

        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    
    def validation_step(self, batch, batch_idx):
        
        list_pred_candidates = []
        list_gold_candidates = []
        
        for text, complex_word, candidates in zip(batch['source'], batch['complex_word'], batch['candidates']):
            list_gold_candidates.append(json.loads(candidates))     
            pred_sents, pred_candidates = generate(text, self.model, self.tokenizer, self.hparams.max_seq_length)
            pred_candidates = remove_word_from_list(complex_word, pred_candidates) # remove pred candidates the same as complex word
            list_pred_candidates.append(pred_candidates)
            
        # acc1 = accuracy_at_k_at_top_gold_1(list_pred_candidates, list_gold_candidates, k=1)
        acc1 = accuracy_at_1(list_pred_candidates, list_gold_candidates)
        
        # print(f'Valid, Acc1: {acc1}')
        
        val_loss = - acc1 
        print(f'val_loss {val_loss}') 
        self.log('val_loss', val_loss, on_step=True, prog_bar=True, logger=True)
       
        return val_loss
    
    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()
        self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = TrainDataset(dataset=self.hparams.dataset, 
                                     tokenizer=self.tokenizer,
                                     preprocessor=self.preprocessor, 
                                     max_len=self.hparams.max_seq_length) 
        
        dataloader = DataLoader(train_dataset,
                                batch_size=self.hparams.train_batch_size,
                                drop_last=True,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4)
        t_total = ((len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                   * self.hparams.gradient_accumulation_steps
                   * float(self.hparams.num_train_epochs)
                   )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        
        val_dataset = EvalDataset(dataset=self.hparams.dataset, preprocessor=self.preprocessor, phase='valid')
        dataloader = DataLoader(val_dataset, 
                          batch_size=self.hparams.valid_batch_size, 
                          drop_last=True,
                          pin_memory=True,
                          shuffle=False,
                          num_workers=4)
        
        return dataloader
        

class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, preprocessor, max_len=128):

        self.preprocessor = preprocessor
        self.data = preprocessor.load_preprocessed_data(dataset, 'train')
        # print(self.data)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        row = self.data.iloc[index]
        complex_sent = row['complex']
        simple_sent = row['simple']
        source = f'simplify: {complex_sent}'

        tokenized_inputs = self.tokenizer(
            [source],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [simple_sent],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze
        
        # source_ids = tokenized_inputs["input_ids"].to(self.device)
        # target_ids = tokenized_targets["input_ids"].to(self.device)

        # src_mask = tokenized_inputs["attention_mask"].to(self.device)
        # target_mask = tokenized_targets["attention_mask"].to(self.device)

        return {'source_ids': source_ids, 
                'source_mask': src_mask, 
                'target_ids': target_ids, 
                'target_mask': target_mask, 
                'source': source, 
                'target': simple_sent}


class EvalDataset(Dataset):
    def __init__(self, dataset, preprocessor, phase):
        self.preprocessor = preprocessor
        self.data = self.preprocessor.load_preprocessed_data(dataset, phase)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # print(row)
        source = self.preprocessor.encode_sentence(row['text'], row['complex_word'])
        # print(f'source: {source}')
        
        return {'source': source, 
                'complex_word': row['complex_word'], 
                'complex_word_index': row['complex_word_index'], 
                'candidates': row['candidates']}

