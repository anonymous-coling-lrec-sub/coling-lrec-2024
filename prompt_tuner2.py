import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

import t5_dataset
from itertools import cycle
from copy import deepcopy
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import matthews_corrcoef, f1_score
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# os.environ["HF_DATASETS_OFFLINE"]= "1"


from datasets import DownloadConfig
import evaluate
dm = DownloadConfig
dm.local_files_only=True
global BLEU
BLEU = evaluate.load("bleu",download_config=dm)
global ROUGE
ROUGE = evaluate.load('rouge',download_config=dm)



class ResMLP(torch.nn.Module):
    def __init__(self,
                 bottleneck_size,
                 module_type='MLP1',
                 emb_dimension=512,
                 nonlinearity='relu', # activation function
                 layer_norm=True,
                 dropout=0.0,
                 residual=False,
                 ):
        """MLP class for soft prompt re-parameterization. MLP can have a Residual connection.
        Args:
            bottleneck_size (int): Dimension of the MLP bottlenack.
            module_type (str, optional): Type of MLP to be used.
                Currently supports 1-layer and 2-layer MLPs, and simple transformer layer ('MLP1'/'MLP2'/'transformer').
                Defaults to 'MLP1'.
            emb_dimension (int, optional): Dimension of T5 model embeddings. Defaults to 512 (T5-small embedding dimension).
            residual (bool, optional): Whether to use residual connection in MLP. Defaults to True.
        """
        super().__init__()
       
        assert module_type in ['MLP1', 'MLP2', 'transformer', 'LSTM', 'LSTM1', 'LSTM2']
        assert nonlinearity in ['relu', 'tanh', 'sigm']

        self.module_type = module_type

        if module_type not in ['LSTM', 'LSTM1', 'LSTM2', 'transformer']:
            layers = [nn.Linear(emb_dimension, bottleneck_size)]

            if nonlinearity=='relu':
                layers.append(nn.ReLU())
            elif nonlinearity=='tanh':
                layers.append(nn.Tanh())
            elif nonlinearity=='sigm':
                layers.append(nn.Sigmoid())

            layers.append(nn.Linear(bottleneck_size, emb_dimension))

            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            if layer_norm:
                layers.append(nn.LayerNorm(emb_dimension))

            if module_type=='MLP2':
                layers = layers + layers # repeat twice
            self.module = torch.nn.Sequential(*layers)

        elif module_type in ['LSTM1', 'LSTM2', 'LSTM']:
            self.lstm_head = torch.nn.LSTM(input_size=emb_dimension,
                                           hidden_size=emb_dimension // 2,
                                           num_layers=1 if module_type=='LSTM1' else 2,
                                           dropout=0.05,
                                           bidirectional=True,
                                           batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(emb_dimension, emb_dimension),
                                          nn.ReLU(),
                                          nn.Linear(emb_dimension, emb_dimension))


        elif module_type=='transformer':
            device = 'cuda'
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dimension, nhead=2, dropout=0.05).to(device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(device)

        self.residual = residual
        if self.residual:
            print('Using skip connection in MLP')

    def forward(self, inputs):
        if self.module_type=='LSTM':
            output_embeds = self.mlp_head(self.lstm_head(inputs)[0]).squeeze()
        elif self.module_type in ['LSTM1', 'LSTM2']:
            output_embeds = self.lstm_head(inputs)[0].squeeze()
            if self.residual:
                output_embeds += inputs
            return output_embeds

        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


class PromptTuner:
    def __init__(self,
                model_name,
                task,
                batch_size=8,
                select_k_per_class=-1,
                prefix_len=0,
                prefix_path=None, # path to the pre-trained progressive prompt
                freeze_weights=True,
                freeze_except='shared',
                lr=0.3,
                weight_decay=1e-5,
                seq_len=512,
                early_stopping=False,
                random_prompt_init=False, # init prompt from random uniform
                prefix_MLP='None',
                residual=True,
                bottleneck_size=800, # bottleneck size in case of using MLP reparametrization
                mlp_dropout=0.0,
                mlp_lr=None,
                separate_mlps=True,
                mlp_layer_norm=False,
                weight_decay_mlp=None,
                get_test_subset=True,
                nonlinearity=None,
                **kwargs):
      
        self.args = kwargs['args']
        self.prefix_len = prefix_len
        self.freeze_weights = freeze_weights
        self.training_mode = kwargs['training_mode']
        self.model_name = model_name # e.g. "t5-large"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.peft_training_config = self.set_training_mode_configs()
    
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # Freezing model weights for prompt tuning
        if self.freeze_weights:

            print('Freezing weights')
            self.do_freeze_weights(except_condition=freeze_except)

        if self.peft_training_config!=None:
            self.model = get_peft_model(self.model, self.peft_training_config)
            self.model.print_trainable_parameters()
       
        """Class for CL & prompt tuning experiments with T5 model.
        Args:
        """
        self.cut = kwargs['cut']

        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'wsc_bool', 'cb', 'record', 'multirc', 'rte_superglue']
        self.task_to_target_len = self.task_to_target_len = {
            'rte': 5,
            'mrpc': 5,
            'sst2': 2,
            'sst': 2,
            'qqp': 5,
            'cola': 5,
            'qnli': 5,
            'mnli': 5,
            'stsb': 3,

            'wic': 2,
            'boolq': 2,
            'copa': 2,
            'wsc': 3,
            'wsc_bool': 2,
            'cb': 5,
            'multirc': 5,
            'record': 10,
            'rte_superglue': 5,

            'imdb': 2,

            'ag_news': 2,
            'yahoo_answers_topics': 5,
            'dbpedia_14': 5,
            'amazon': 2,
            'yelp_review_full': 2,
            'cnndm': 128,
            'xsum': 128,
            'dart': 128,
            'enro': 128,
            'snli': 5,
        
        }
        self.task = task

        
        # self.lr = lr
        self.set_lr(self.task)
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.select_k_per_class = select_k_per_class
        self.early_stopping = early_stopping

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


        
        self.prefix_path = prefix_path
        # Creating a trainable soft prompt
        if self.prefix_len>0:
            self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len, random_init=random_prompt_init),
                                                          requires_grad=True))

        print("head requires_grad",self.model.lm_head.weight.requires_grad)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("total # parameters:",pytorch_total_params)

        # exit()
        # Model to cuda
        self.model.to(self.device)
        # Create MLP (if prompt re-parameterization is requested)
        self.residual=residual
        self.separate_mlps=separate_mlps
        if self.prefix_len <0:
            prefix_MLP='None'
        self.get_MLP(prefix_MLP, bottleneck_size, mlp_layer_norm, mlp_dropout,
                     separate_mlps=separate_mlps) # adds prompt MLP reparametrization (and puts to cuda)

        self.weight_decay = weight_decay
        self.mlp_lr = mlp_lr
        self.weight_decay_mlp = weight_decay_mlp
        self.optimizer = self.get_optimizer(lr, weight_decay,
                                            task=self.task,
                                            mlp_lr=mlp_lr,
                                            weight_decay_mlp=weight_decay_mlp)

        # Create best prompt/model copy for early stopping
        if self.early_stopping:
            if self.prefix_len>0:
                # prompt tuning
                self.best_prompt = self.model.prompt.detach().cpu().numpy()
            else:
                # model tuning
                self.best_model = deepcopy(self.model.state_dict()) # saving best model
            self.best_acc = 0.0 # best avg accuracy on seen tasks

        # Get task -> data dictionary for CL training
        self.get_test_subset = get_test_subset
        self.tasks_data_dict = self.get_tasks_data_dict()


    # Create optimizer
    def set_lr(self,task):
        mode = self.training_mode
        if task in ['snli','rte','sst','sst2','cola']:

            lr = {
                'lora':1e-2,
                'bf':3e-1,
                'fine_tune':3e-4,
                'last_layer':3e-2,
                    }
        else:
            lr = {
                'lora':2e-4,
                'bf':3e-1,
                'fine_tune':3e-5,
                'last_layer':3e-2,
                    }
        self.lr = lr[mode]

    def set_training_mode_configs(self):
        tm = self.training_mode
        prev_prefix_len = self.prefix_len
        self.prefix_len = -1
        self.freeze_weights = True
        if tm == 'lora':
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=4, lora_alpha=32, lora_dropout=0.0)
            return peft_config
        elif tm == 'last_layer':
            self.prefix_len = -1
            self.freeze_weights = False
            condition = ["lm_head"]
            self.do_freeze_weights(except_condition=condition)
            self.model.lm_head.weight.requires_grad=True

        elif tm == "adapter":
            pass
        elif tm =="fine_tune":
            self.freeze_weights = False
            condition = '.'
            self.do_freeze_weights(except_condition=condition)

        elif tm =="bf":
            self.freeze_weights = True
            self.prefix_len = prev_prefix_len
            
        return None


    def get_optimizer(self, lr, weight_decay,
                      task=None, mlp_lr=None, weight_decay_mlp=None): # task is used for MLP

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },

            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
        ]

        if task!=None and self.prefix_MLP!=None:
            if weight_decay_mlp==None:
                weight_decay_mlp = weight_decay
            if mlp_lr==None:
                mlp_lr = lr

            if self.separate_mlps==False: # shared MLP
                all_mlps = [self.prefix_MLP]
            else: # separate MLP for each token
                all_mlps = [self.prefix_MLP[i] for i in list(self.prefix_MLP)]

            for mlp in all_mlps:
                optimizer_grouped_parameters.append({
                    "params": [p for n, p in mlp.named_parameters()],# if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay_mlp,
                    "lr": mlp_lr,
                })
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        print(sum(p.numel() for p in mlp.parameters()))
        return optimizer


    # Create MLP for prompt tuning
    def get_MLP(self, prefix_MLP, bottleneck_size, layer_norm, mlp_dropout, separate_mlps=False):
        if prefix_MLP == 'None':
            self.prefix_MLP = None
        else:
            print('Using MLP reparametrization with bottleneck = ', bottleneck_size)
            N = self.model.encoder.embed_tokens.weight.shape[1]

            if separate_mlps: # separate MLP for each prompt token
                print('Creating a dictionary of MLPs')
                m = self.prefix_len
                self.prefix_MLP = {}
                for i in range(m):
                    self.prefix_MLP[i] = ResMLP(bottleneck_size=bottleneck_size,
                                         module_type=prefix_MLP,
                                         dropout=mlp_dropout,
                                         emb_dimension=N,
                                         nonlinearity='relu', # activation function
                                         layer_norm=layer_norm,
                                         #residual=True
                                         residual=self.residual,
                                         )
                for i in list(self.prefix_MLP):
                    self.prefix_MLP[i].to(self.device)

            else: # 1 shared MLP
                self.prefix_MLP = ResMLP(bottleneck_size=bottleneck_size,
                                         module_type=prefix_MLP,
                                         dropout=mlp_dropout,
                                         emb_dimension=N,
                                         nonlinearity='relu', # activation function
                                         layer_norm=layer_norm,
                                         #residual=True
                                         residual=self.residual,
                                         )
                self.prefix_MLP.to(self.device)
        #if self.prefix_MLP!=None:
        #    self.prefix_MLP.to(self.device)
    # 1 sample 
    # 2 sample
    
    # Initialize new task prompt from random vocab. tokens
    def init_new_prompt(self, prompt_len, random_init=False):
        if self.prefix_path==None:
            model = self.model
            N = model.encoder.embed_tokens.weight.shape[0]
            prompt_weigths = []

            # init from random uniform
            if random_init:
                r1, r2 = -0.5, 0.5
                x = (r1 - r2) * torch.rand(prompt_len, N) + r2
                prompt_weigths = x.numpy()
            else: # init from random tokens
                for i in range(prompt_len):
                    with torch.no_grad():
                        j = np.random.randint(N) # random token
                        w = deepcopy(model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                        prompt_weigths.append(w)
            prompt_weigths = np.array(prompt_weigths)

        else: # initializing from existing path
            prompt_weigths = np.load(self.prefix_path)
        return prompt_weigths


    # passing each prompt token through a separate MLP
    def pass_separate_mlps(self):
        x = self.model.prompt
        out = []
        for i in range(self.prefix_len):
            h = self.prefix_MLP[i](x[i:i+1])
            out.append(h)
        out = torch.concat(out)
        return out


    # Create Dictionary of task_name -> dataloader (for CL experiments)
    def get_tasks_data_dict(self):
        tasks_data_dict = {}

        task = self.task
        tasks_data_dict = {}

        data_params = {'task': task,
                       'batch_size': self.batch_size,
                       'max_length': self.seq_len,
                       'target_len': self.task_to_target_len[task],
                       'prefix_list': [], # we are using vector prefix (instead of tokenization)
                       }
        ds2 = t5_dataset.TemplateDatasetForClassification(tokenizer=self.tokenizer, task=task)
       
        if task in ['rte','cola','sst']:
            k = self.select_k_per_class
            k_val = -1
        elif task in ['cnndm','xsum','dart','enro','snli']:
            k = self.select_k_per_class
            k_val = 1000
            
        else:
            k = self.select_k_per_class if (self.select_k_per_class<=500 and task not in ['cb', 'copa', 'wsc', 'wsc_bool']) else -1
            k_val = -1
        # if self.get_test_subset==False: k_val = -1 # use all val set
      
        dataloader_train = ds2.get_final_ds(**data_params, k=k, split='train',cut=self.cut)
        print('k = ', k, '  k-val = ',k_val)
        val_split = 'val' #if (task in self.glue_datasets) or (task in self.superglue_datasets) else 'test'
        
        dataloaders = ds2.get_final_ds(**data_params, k=k_val,
                                       split=val_split, return_test=self.get_test_subset,cut=self.cut)

        tasks_data_dict['train'] = dataloader_train

        if self.get_test_subset:
            dataloader_val, dataloader_test = dataloaders[0], dataloaders[1]
            tasks_data_dict['val'] = dataloader_val
            tasks_data_dict['test'] = dataloader_test
        else:
            tasks_data_dict['val'] = dataloaders

        if task == 'multirc' and k_val==-1:
            self.multirc_idx = ds2.multirc_idx # saving multirc idx for later computation
        else: self.multirc_idx = None

        return tasks_data_dict


    # Perform one train step for prompt tuning (following Lester et al.)
    def train_step_lester(self,
                          batch,
                          task=None,
                          get_pred=False):
        prefix_len = self.prefix_len
        model = self.model
        embed_prompt = self.prefix_MLP!=None
        if embed_prompt:
            mlp = self.prefix_MLP
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])

        k = inputs_embeds.shape[0]
        if embed_prompt:
            if self.separate_mlps==False:
                prompt = mlp(model.prompt)
           
            else:
                prompt = self.pass_separate_mlps()
        else:
            prompt = model.prompt

        inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                      inputs_embeds], axis=1)[:,:self.seq_len]
        full_prefix_len = prompt.shape[0]

        source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                             batch["source_mask"]), axis=1)[:,:self.seq_len]

        encoder_outputs = model.encoder(
                                attention_mask=source_mask_updated,
                                inputs_embeds=inputs_embeds,
                                head_mask=None,
                                output_attentions=None,
                                output_hidden_states=None,
                                return_dict=None,
                            )

        if get_pred:
            assert task!=None
            outs = model.generate(
                input_ids=batch["source_ids"],
                attention_mask=source_mask_updated,
                encoder_outputs=encoder_outputs,
                max_length=self.task_to_target_len[task],
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]
            targets = [self.tokenizer.decode(ids) for ids in batch['target_ids']]
            row_true, row_pred = self.preprocess_outputs(task, dec, targets)
        else: row_true, row_pred = None, None


        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=source_mask_updated,
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0]

        return loss, row_true, row_pred



    # Perform one train step for full model training
    def train_step(self, batch, task=None, get_pred=False):
        model = self.model
        tokenizer = self.tokenizer

        batch = {k: batch[k].to(self.device) for k in batch}

        inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        encoder_outputs = model.encoder(
                                #input_ids=batch["source_ids"],
                                attention_mask=batch["source_mask"],
                                #labels=lm_labels,
                                #decoder_attention_mask=batch['target_mask']
                                #input_ids=input_ids,
                                #attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds,
                                head_mask=None, #head_mask,
                                output_attentions=None, #output_attentions,
                                output_hidden_states=None, #output_hidden_states,
                                return_dict=None, #return_dict,
                            )

        if get_pred:
            assert task!=None
            outs = model.generate(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                encoder_outputs=encoder_outputs,
                max_length=self.task_to_target_len[task],
            )
            dec = [self.tokenizer.decode(ids) for ids in outs]
            targets = [self.tokenizer.decode(ids) for ids in batch['target_ids']]
            row_true, row_pred = self.preprocess_outputs(task, dec, targets)
        else: row_true, row_pred = None, None

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

        outputs = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            encoder_outputs=encoder_outputs,
        )
        loss = outputs[0]

        return loss, row_true, row_pred



    # Process string for validation (remove pad and end tokens)
    def normalize_text(self, s, punct_to_keep=[]):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the|)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            text2 = text.replace('<pad>', '').replace('</s>', '')
            exclude = list(set(string.punctuation))
            exclude = set([x for x in exclude if x not in punct_to_keep])
            return "".join(ch for ch in text2 if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    # Compute EM score used for some SuperGLUE tasks
    def compute_exact_match(self, prediction, truth):
        return int(self.normalize_text(prediction) == self.normalize_text(truth))


    # Compute F1 score used for some GLUE & SuperGLUE tasks
    def compute_f1(self, prediction, truth):
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)


    # Normalize texts
    def preprocess_outputs(self,
                           task,
                           decoded, # T5 decoded output
                           targets, # target output
                           ):
        if task=='record':
            punct_to_keep = [';'] # we need ; to separate possible answers
        else:
            punct_to_keep = []

        row_true = [self.normalize_text(x, punct_to_keep) for x in targets]
        row_pred = [self.normalize_text(x, punct_to_keep) for x in decoded]

        if task=='stsb':
            # convert digits to float, convert letters to 0
            row_true = [float(x) if any(c.isalpha() for c in x)==False else 0.0 for x in row_true]
            row_pred = [float(x) if any(c.isalpha() for c in x)==False else 0.0 for x in row_pred]

        return row_true, row_pred


    # Compute task metrics
    def compute_task_metrics(self, task, y_true, y_pred, training=False):
        if task=='cola':
            return matthews_corrcoef(y_true, y_pred)

        elif task=='stsb':
            return np.corrcoef(y_true, y_pred)[0,1]

        elif task=='cb':
            return np.mean(np.array(y_true) == np.array(y_pred)), f1_score(y_true, y_pred, average='macro')

        elif task=='multirc':
            if (self.multirc_idx is None) == False and (training==False): # we have stored multirc idx & NOT training
                em = []
                for idx in set(self.multirc_idx):
                    k = np.where(self.multirc_idx==idx)[0]
                    score = (np.array(y_true)[k] == np.array(y_pred)[k]).all()
                    em.append(score)
                return np.mean(em), f1_score(y_true, y_pred, average='micro')
            else:
                return f1_score(y_true, y_pred, average='micro')
        elif task in ['cnndm','xsum']:
            return self.rouge(y_true,y_pred)
        
        elif task in ['dart','enro']:

            return self.bleu(y_true,y_pred)


        else: # computing accuracy / other scores
            if task=='record':
                # multiple answers
                corr, f1, total = 0,0,0
                for x,y in zip(y_pred, y_true):
                    corr += max([self.compute_exact_match(x, yi) for yi in y.split(';')])
                    f1 += max([self.compute_f1(x, yi) for yi in y.split(';')])
                total = len(y_true)
                return corr/total, f1/total
            else:
                # correct predictions
                corr = np.sum([x==y for x,y in zip(y_pred, y_true)])
                total = len(y_true) # total predictions
                return corr/total



    # Compute loss / accuracy on a validation (train / test) set
    def validate(self,
                 dataloader_val,
                 task,
                 prompt=None,
                 target_len=2,
                 print_outputs=False,
                 iter=0,
                 write_path=None
                ):

        model = self.model
        prefix_len = self.prefix_len
        max_length = target_len
        tokenizer = self.tokenizer
        model.eval()

        y_pred, y_true, loss_values = [], [], []
        for i, batch in enumerate(tqdm(dataloader_val)):
            batch = {k:batch[k].to(self.device) for k in batch}
            inputs_embeds = model.encoder.embed_tokens(batch["source_ids"]).to(self.device)
            if prompt!=None:
                k = inputs_embeds.shape[0]
                inputs_embeds = torch.concat([prompt.repeat(k, 1, 1),
                                              inputs_embeds], axis=1)[:,:self.seq_len]

                full_prefix_len = prompt.shape[0] # prompt is inputted by user
                source_mask_updated = torch.concat( (batch["source_mask"][0][0].repeat(k,full_prefix_len),
                                                     batch["source_mask"]), axis=1)[:,:self.seq_len]

            else: # full model fine tuning, no prompt added
                source_mask_updated = batch["source_mask"]


            encoder_outputs = model.encoder(
                                    attention_mask=source_mask_updated,
                                    inputs_embeds=inputs_embeds,
                                    head_mask=None,
                                    output_attentions=None,
                                    output_hidden_states=None,
                                    return_dict=None,
                                )

            outs = model.generate(
                input_ids=batch["source_ids"],
                attention_mask=source_mask_updated,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
            )

            dec = [tokenizer.decode(ids) for ids in outs]
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

            row_true, row_pred = self.preprocess_outputs(task, dec, targets)

            y_true += row_true
            y_pred += row_pred

            lm_labels = batch["target_ids"]
            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            # print("val:")
            # print("true",y_true[0])
            # print("pred",y_pred[0])
            loss = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=lm_labels,
                decoder_attention_mask=batch['target_mask'],
                encoder_outputs=encoder_outputs).loss
            loss_values.append(loss.detach().cpu().numpy())
        score = self.compute_task_metrics(task, y_true, y_pred)
        loss = np.mean(loss_values)
        print("y_true:",y_true[:10])
        print("y_pred:",y_pred[:10])
        print("score")
        if write_path!=None:
            truth_path = f"{write_path}/truth_{iter}.txt" 
            pred_path =  f"{write_path}/pred_{iter}.txt" 
            with open(truth_path,'w') as truth_file:
                with open(pred_path,'w') as pred_file:
                    for t in y_true:
                        truth_file.write(t)
                        truth_file.write('\n')
                    for p in y_pred:
                        pred_file.write(p)
                        pred_file.write('\n')

        return {"score": score, "loss": loss}



    # Freeze model weights
    def do_freeze_weights(self, except_condition='shared'):
        model = self.model
        # freeze all first
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                param.requires_grad = False
            
        for name, param in model.named_parameters():
            
            if isinstance(except_condition,str) and except_condition in name:
                param.requires_grad = True

            elif isinstance(except_condition,list):
                for exc in except_condition:
                    if exc in name:
                        print(name)
                        param.requires_grad = True



#     # Freeze / unfreeze MLPs for given tasks (when requires_grad==False then freezing)
#     def freeze_unfreeze_mlps(self, tasks, requires_grad=False):
#         assert self.prefix_MLPs != None

#         for t in tasks:
#             #for name, param in self.prefix_MLPs[t].named_parameters():
#             for name, param in self.prefix_MLPs[t].named_parameters():
#                 if param.requires_grad != requires_grad:
#                     param.requires_grad = requires_grad
#                     param.grad = None # remove old gradient



    def update_best_model(self, acc):
        if acc>self.best_acc:
            # getting best prompt
            if self.prefix_len>0 and self.separate_mlps==False:
                best_prompt = self.model.prompt
                if self.prefix_MLP!=None:
                    self.prefix_MLP.eval()
                    best_prompt = self.prefix_MLP(best_prompt)

                self.best_prompt = best_prompt.detach().cpu().numpy()

            # getting best model
            else:
                self.best_model = deepcopy(self.model.state_dict()) # saving best model
            self.best_acc = acc # best avg accuracy on seen tasks



    # Perform training on a single task
    def train_one_task(self,
                       epochs=40,
                       eval_every_N=1,
                       save_path=''):
        task = self.task
        if epochs == 0:
            target_len = self.task_to_target_len[task]
            dataloader_val = self.tasks_data_dict['val']
            acc_dict = self.validate(dataloader_val, task,
                                            prompt=None, target_len=target_len, print_outputs=True)
            return acc_dict
        
        print('task = ', task)
        if self.early_stopping:
            self.best_acc = 0.0 # re-setting best acc

        if self.prefix_MLP!=None:
            print('Freezing all MLPs except for ', task)
            mlp = self.prefix_MLP

        model = self.model

        with torch.no_grad():
            model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
                                        requires_grad=True))
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay, task=task)
        model.to(self.device)
        target_len = self.task_to_target_len[task]
        dataloader_train = self.tasks_data_dict['train']
        dataloader_val = self.tasks_data_dict['val']

        score_dict = {"val":   {"acc": [], "loss": []},
                      "train": {"acc": [], "loss": []}}

        loss_train = []
        best_acc = -1
        for epoch in range(epochs):
            print("epoch:",epoch)
            model.train()
            if self.prefix_MLP!=None:
                if self.separate_mlps:
                    for j in list(self.prefix_MLP):
                        self.prefix_MLP[j].train()
                else:
                    mlp.train()

            y_pred, y_true = [], [] # to compute train acc
            
            for i, batch in enumerate(tqdm(dataloader_train)):
                batch = {k:batch[k].to('cuda') for k in batch}
                get_pred = False #i<25)
                if self.prefix_len>0: # prompt tuning
                    loss, row_true, row_pred = self.train_step_lester(batch,
                                                                      task=task,
                                                                      get_pred=get_pred,
                                                                     )
                else: # fine-tuning
                    loss, row_true, row_pred = self.train_step(batch, task=task, get_pred=get_pred)
                loss_train.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if get_pred: # we compute train score on first 250 batches to speed up computation
                    y_true += row_true
                    y_pred += row_pred
            # if get_pred:
            #     print("train: ")
            #     print("true",y_true[0])
            #     print("pred",y_pred[0])
            if get_pred:
                score_dict['train']['acc'].append( self.compute_task_metrics(task, y_true, y_pred, training=True) )
            score_dict['train']['loss'].append(np.mean(loss_train))
            loss_train = []
            # evaluate accuracy after each epoch
            if self.prefix_MLP!=None:
                if self.separate_mlps:
                    for j in list(self.prefix_MLP):
                        self.prefix_MLP[j].eval()
                    prompt = self.pass_separate_mlps()
                else:
                    mlp.eval()
                    prompt = mlp(model.prompt)
            else:
                if self.prefix_len>0:
                    prompt = model.prompt
                else:
                    prompt = None

            if epoch%eval_every_N == 0:
                acc_dict = self.validate(dataloader_val, task,
                                         prompt=prompt, target_len=target_len, print_outputs=True)
#                 if task in ['record', 'cb'] or (task=='multirc' and self.multirc_idx!=None):
#                     acc = np.mean(acc) # averaging 2 scores
#                 val_acc.append(acc)       
                
                if task in ["cnndm","xsum"]:
                    acc = acc_dict['score']['rougeL']
                     # averaging in case we have 2 metrics
                elif task in ['dart','enro']:
                    acc = acc_dict['score']['bleu']
                else:
                    acc = np.mean(acc_dict['score'])
                score_dict['val']['acc'].append(acc_dict['score'])
                score_dict['val']['loss'].append(acc_dict['loss'])

                print(epoch, task, '->', score_dict['val'])
                if self.early_stopping:
                    self.update_best_model(acc)
                
                if save_path!='':
                    if acc>=best_acc:
                        
                        best_acc = acc
                        save_path2 = f"{self.args.save_dir}/{self.task}/{self.args.model_name}/{self.training_mode}"
                        model_path = f"{save_path2}/{self.args.db_name}"
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        print(f'saving model to {model_path}')
                        with open(f"{model_path}/log.txt","w",encoding='utf-8') as result_out:
                                result_out.write(f"{best_acc}")
                                if task in ["cnndm","xsum",'dart','enro']:
                                    result_out.write(f"\t{acc_dict['score']}")
                                
                                result_out.write("\n")
                        np.save(os.path.join(model_path, 'best_prompt.npy'), model.prompt.detach().cpu().numpy())
                        np.save(os.path.join(model_path, 'score_dict.npy'), score_dict)
                    np.save(os.path.join(save_path, 'score_dict.npy'), score_dict)
                    print(f'saving model at {model_path}')
#         if self.early_stopping:
#             self.restore_best_model()

        return score_dict


    def rouge(self,y_true,y_pred):
        # rouge = evaluate.load('rouge')
        y_true_list = [[y] for y in y_true]
        result = ROUGE.compute(predictions=y_pred,references=y_true_list)
        return result
    
    def bleu(self,y_true,y_pred):
        # BLEU = evaluate.load("bleu")
        """
        predictions = [
            ["hello there general kenobi",
            ["foo bar foobar"]
        ]
        references = [
            [["hello there general kenobi"], ["hello there!"]],
            [["foo bar foobar"]]
        ]
        """
        
        y_true_list = [[y] for y in y_true]
        result = BLEU.compute(predictions=y_pred,references=y_true_list)
        return result

