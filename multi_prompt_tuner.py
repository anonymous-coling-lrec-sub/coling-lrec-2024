from prompt_tuner2 import PromptTuner,ResMLP
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
import evaluate
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType



class MultiClassTuner(PromptTuner):
    def __init__(self,
    *args, 
    **kwargs):

        super(MultiClassTuner,self).__init__(*args, **kwargs)
        # print(kwargs['args'])
        args = kwargs['args']
      
        self.task_type = args.task_type
        self.val_task = args.val_task
        if args.lr > 0:
            self.lr = args.lr
        else:
            self.set_lr(self.task_type)
        print(self.lr)

    def set_lr(self,task_type):
        mode = self.training_mode
        if task_type in ['cls']:

            lr = {
                'lora':2e-4,
                'bf':7e-1,
                'fine_tune':3e-4,
                'last_layer':3e-2,
                    }
        else:
            lr = {
                'lora':2e-4,
                'bf':3e-2,
                'fine_tune':3e-4,
                'last_layer':3e-2,
                    }
        self.lr = lr[mode]

    def get_multi_tasks_data_dict(self,task_type='cls'):
        multi_task_dict = {}
        cls_tasks = ["cola", "rte", "sst", "snli"]
        gen_tasks = [ "cnndm", "xsum", "dart", "enro"]
        if self.task_type == 'cls':
            tasks = cls_tasks
        else:
            tasks = gen_tasks
        print(tasks)
        for task in tasks:      
            task_dict = self.get_tasks_data_dict(task)
      

            multi_task_dict[task] = task_dict
        
        return multi_task_dict

    def get_tasks_data_dict(self,task=None):
        if task == None:
            return 
        tasks_data_dict = {}
        if task is None:
            task = self.task
        data_params = {'task': task,
                       'batch_size': self.batch_size,
                       'max_length': self.seq_len,
                       'target_len': self.task_to_target_len[task],
                       'prefix_list': [], # we are using vector prefix (instead of tokenization)
                       }
        ds2 = t5_dataset.T5Dataset(self.tokenizer, task)
        ds2 = t5_dataset.TemplateDatasetForClassification(tokenizer=self.tokenizer, task=task)

        if task in ['rte','cola','sst']:
            k = self.select_k_per_class
        elif task in ['cnndm','xsum','dart','enro','snli']:
            k = self.select_k_per_class
            
        else:
            k = self.select_k_per_class if (self.select_k_per_class<=500 and task not in ['cb', 'copa', 'wsc', 'wsc_bool']) else -1
            k_val = -1
        # if self.get_test_subset==False: k_val = -1 # use all val set
      
        dataloader_train = ds2.get_multi_final_ds(**data_params, k=k, split='train')
        print('k = ', k, '  k-val = ',k_val)
        val_split = 'validation' if (task in self.glue_datasets) or (task in self.superglue_datasets) else 'test'
        
        dataloaders = ds2.get_multi_final_ds(**data_params, k=k_val,
                                       split=val_split, return_test=self.get_test_subset)

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


    def train_multi_task(self,
                       epochs=40,
                       eval_every_N=1,
                       save_path='',
                       val_task='re'):
        save_path2 = f"{self.args.save_dir}/{self.val_task}/{self.args.model_name}"
        model_path = f"{save_path2}/{self.training_mode}"
        cls_tasks = ["cola", "rte", "sst", "snli"]
        gen_tasks = ["cnndm", "dart", "enro","xsum"]
        tasks = self.get_multi_tasks_data_dict()
        train_data = {}
        for k in tasks:
            if k == self.val_task:
                val_data =  {k:tasks[k]['val']}
            else:
                train_data[k] = tasks[k]['train']
        
        if self.early_stopping:
            self.best_acc = 0.0 # re-setting best acc

        if self.prefix_MLP!=None:
            print('Freezing all MLPs except for ', tasks)
            mlp = self.prefix_MLP

        model = self.model

        with torch.no_grad():
            model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.prefix_len),
                                        requires_grad=True))
            self.optimizer = self.get_optimizer(self.lr, self.weight_decay, task=tasks)
        model.to(self.device)

        # data stuff
        # target_len = self.task_to_target_len[task]
        # dataloader_train = self.tasks_data_dict['train']
        # dataloader_val = self.tasks_data_dict['val']


        score_dict = {"val":   {"acc": [], "loss": []},
                      "train": {"acc": [], "loss": []}}

        loss_train = []
        best_acc = -1
        for epoch in range(epochs):
            print(epoch)
            model.train()
            if self.prefix_MLP!=None:
                if self.separate_mlps:
                    for j in list(self.prefix_MLP):
                        self.prefix_MLP[j].train()
                else:
                    mlp.train()

            y_pred, y_true = [], [] # to compute train acc
            for task,dataloader_train in train_data.items():#something:
                print(task)
                target_len = self.task_to_target_len[task]
                # dataloader_train = each_dataloader_train['train']
                # dataloader_val = each_dataloader_train['val']

                # if task != cls_tasks[-1] or task != gen_tasks[-1]:
 
                for i, batch in enumerate(tqdm(dataloader_train)):
                    batch = {k:batch[k].to('cuda') for k in batch}
                    
                    if self.prefix_len>0: # prompt tuning
                        loss, row_true, row_pred = self.train_step_lester(batch,
                                                                            task=task,
                                                                            #task=task if self.prefix_MLP!=None else None,
                                                                            get_pred=i<250,
                                                                            )
                    else: # fine-tuning
                        loss, row_true, row_pred = self.train_step(batch, task=task, get_pred=i<250)
                    loss_train.append(loss.detach().cpu().numpy())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if i<250: # we compute train score on first 250 batches to speed up computation
                        y_true += row_true
                        y_pred += row_pred

            print("train: ")
            print("true",y_true[0])
            print("pred",y_pred[0])
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
                dataloader_val = val_data[val_task]
                val_target_len = self.task_to_target_len[val_task]
                
                
                acc_dict = self.validate(dataloader_val, val_task,
                                         prompt=prompt, target_len=val_target_len, print_outputs=True,iter=0,write_path=model_path)
                print("acc_dict:::",val_task,acc_dict)
#                 if task in ['record', 'cb'] or (task=='multirc' and self.multirc_idx!=None):
#                     acc = np.mean(acc) # averaging 2 scores
#                 val_acc.append(acc)       
                
                if val_task in ["cnndm","xsum"]:
                    acc = acc_dict['score']['rougeL']
                     # averaging in case we have 2 metrics
                elif val_task in ['dart','enro']:
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
                        save_path2 = f"{self.args.save_dir}/{self.val_task}/{self.args.model_name}"
                        model_path = f"{save_path2}/{self.training_mode}"
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        with open(f"{model_path}/log_{str(self.lr)}.txt","w",encoding='utf-8') as result_out:
                                result_out.write(f"{best_acc}")
                                if task in ["cnndm"]:
                                    result_out.write(f"{acc_dict['score']}")
                                result_out.write("\n")
                        np.save(os.path.join(model_path, f'best_prompt_{str(self.lr)}.npy'), model.prompt.detach().cpu().numpy())
                        np.save(os.path.join(model_path, f'score_dict_{str(self.lr)}.npy'), score_dict)
                    np.save(os.path.join(save_path, f'score_dict_{str(self.lr)}.npy'), score_dict)
#         if self.early_stopping:
#             self.restore_best_model()

        return score_dict