import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from datasets import Dataset as hf_Dataset
import datasets
import lm_eval
from lm_eval import tasks, evaluator
from promptsource.promptsource.templates import DatasetTemplates
import fnmatch
from transformers import T5Tokenizer
from dart_process import process_triples

class T5Dataset:
    def __init__(self, tokenizer, task, padding='max_length'):
        self.tokenizer = tokenizer
        self.glue_datasets = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', \
                              'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']
        self.superglue_datasets = ['copa', 'boolq', 'wic', 'wsc', 'cb', 'record', 'multirc', 'rte_superglue', 'wsc_bool']
        self.task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            #"qnli": ("question", "sentence"),
            "qnli": ("text1", "text2"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),

            "boolq": ("passage", "question"),
            "copa": ('choice1', 'choice2', 'premise', 'question'),
            "wic": ("start1", "end1", "sentence1", "start2", "end2", "sentence2", "word"),
            "wsc": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
            "wsc_bool": ("span1_text", "span1_index", "span2_text", "span2_index", "text"),
            "cb": ("premise", "hypothesis"),
            "record": ("passage", "query", "entities"),
            "multirc": ("question", "answer", "paragraph"),
            "rte_superglue": ("premise", "hypothesis"),

            "scicite": ("sectionName", "string"),
            "imdb": ("text", None),

            "ag_news": ("text", None),
            "yelp_review_full": ("text", None),
            "yahoo_answers_topics": ("question_content", "best_answer"),
            "dbpedia_14": ("title", "content"),

            "ag": ("content", None),
            "yelp": ("content", None),
            "yahoo": ("content", None),
            "dbpedia": ("content", None),
            "amazon": ("content", None),
        }

        self.task_to_labels = {
            "cola": ("not_acceptable", "acceptable"),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mnli-mm": (),
            "mrpc": ("not_equivalent", "equivalent"),
            "qnli": ("entailment", "not_entailment"),
            "qqp": ("not_duplicate", "duplicate"),
            "rte": ("entailment", "not_entailment"),
            "sst2": ("negative", "positive"),
            "stsb": (),
            "wnli": (),

            "boolq": ("false", "true"),
            "copa": ("false", "true"),
            "wic": ("false", "true"),
            "wsc_bool": ("false", "true"),
            "cb": ("entailment", "contradiction", "neutral"),
            "multirc": ("false", "true"),
            "rte_superglue": ("entailment", "not_entailment"),

            "scicite": (),
            "imdb": ("negative", "positive"),

            "ag_news": ("world", "sports", "business", "science"),
            "yelp_review_full": ("terrible", "bad", "middle", "good", "wonderful"),
            "yahoo_answers_topics": ("society and culture", "science", "health", "education and reference",
                                     "computers and internet", "sports", "business", "entertainment and music",
                                     "family and relationships", "politics and government"),
            "dbpedia_14": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                           "meanoftransportation", "building", "naturalplace", "village", "animal",
                           "plant", "album", "film", "writtenwork"),

            "ag": ("world", "sports", "business", "science"),
            "yelp": ("terrible", "bad", "middle", "good", "wonderful"),
            "yahoo": ("society and culture", "science", "health", "education and reference",
                      "computers and internet", "sports", "business", "entertainment and music",
                      "family and relationships", "politics and government"),
            "dbpedia": ("company", "educationalinstitution", "artist", "athlete", "officeholder",
                        "meanoftransportation", "building", "naturalplace", "village", "animal",
                        "plant", "album", "film", "writtenwork"),
            "amazon": ("terrible", "bad", "middle", "good", "wonderful"),
        }

        self.task = task
        self.padding = padding
        self.label_key = 'label'
        if 'yahoo_' in task: self.label_key = 'topic'
        if 'stsb' in task: self.label_key = 'similarity_score'
        if task=='record': self.label_key = 'answers'


    def save_multirc_questions_idx(self, val_ds):
        idx = []
        i = 0
        x_prev, y_prev= val_ds['paragraph'][0], val_ds['question'][0]

        for x,y in zip(val_ds['paragraph'], val_ds['question']):
            if x_prev!=x or y_prev!=y:
                i += 1
            x_prev = x
            y_prev = y
            idx.append(i)
        self.multirc_idx = np.array(idx)


    def select_subset_ds(self, ds, k=2000, seed=0):
        if self.task in ['stsb', 'record', 'wsc']: # non-discrete labels
            idx_total = np.random.choice(np.arange(ds.shape[0]), min(k,ds.shape[0]), replace=False)

        else:
            label_key = self.label_key
            N = len(ds[label_key])
            idx_total = np.array([], dtype='int64')

            for l in set(ds[label_key]):
                idx = np.where(np.array(ds[label_key]) == l)[0]
                idx_total = np.concatenate([idx_total, # we cannot take more samples than there are available
                                            np.random.choice(idx, min(k, idx.shape[0]), replace=False)])

        np.random.seed(seed)
        np.random.shuffle(idx_total)
        return ds.select(idx_total)


    def process_wsc(self, wsc_row):
        text_proc = wsc_row['text'].split(' ')
        #text_proc[wsc_row['span1_index']] = '*' + text_proc[wsc_row['span1_index']] +'*'
        target = text_proc[wsc_row['span1_index']]
        text_proc[wsc_row['span2_index']] = '*' + text_proc[wsc_row['span2_index']] + '*'
        text_proc = (' ').join(text_proc)
        return text_proc, target


    def preprocess_function(self, examples, task,
                            max_length=512, max_length_target=2,
                            prefix_list=[]):
        tokenizer = self.tokenizer
        #key1, key2 = self.task_to_key[task]
        keys = self.task_to_keys[task]
        label_key = self.label_key

        if keys[1]!=None:
            if task=='record':
                text = 'passage : ' + str(examples['passage']) + ' query: ' + str(examples['query']) + ' entities: ' + ('; ').join((examples['entities']))
            elif task=='wsc':
                text, target = self.process_wsc(examples)
            else:
                text = ''
                for key in keys:
                    text += key + ': ' + str(examples[key]) + ' '
        else:
            text = examples[keys[0]]

        if len(prefix_list)>0:
            text = (' ').join(prefix_list) + ' ' + text
        source = tokenizer(text.strip()+' </s>',
                          truncation=True,
                          padding=self.padding,
                          #padding='max_length',
                          max_length=max_length)

        #target = 'positive' if examples['label']==1 else 'negative'
        if task=='stsb':
            target = str(examples[label_key])[:3]
        elif task=='record':
            target = '; '.join(examples[label_key])
        elif task=='wsc':
            pass # already obtained target
        else:
            target = self.task_to_labels[task][examples[label_key]]
        target += ' </s>'
        target = tokenizer(
                  target,
                  max_length=max_length_target,
                  #pad_to_max_length=True,
                  pad_to_max_length=self.padding=='max_length',
                  #return_tensors="pt"
                )

        dict_final = {"source_ids": source['input_ids'],
                      "source_mask": source['attention_mask'],
                      "target_ids": target['input_ids'],
                      "target_mask": target['attention_mask']}
        return dict_final


    def get_final_ds(self, task, split,
                     batch_size,
                     k=-1,
                     seed=0,
                     return_test=False,
                     target_len=2,
                     max_length=512,
                     prefix_list=[],
                     **kwargs):

        if task in ['amazon']: # amazon not available with hugging face
            df = pd.read_csv('../datasets/src/data/'+task+'/'+split+'.csv', header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df['label'] = df['label'] - 1
            dataset = datasets.Dataset.from_pandas(df)
        elif task == 'mnli':
            dataset = load_dataset('LysandreJik/glue-mnli-train', split=split)
        elif task == 'qnli':
            dataset = load_dataset('SetFit/qnli', split=split)
        elif task == 'stsb':
            dataset = load_dataset('stsb_multi_mt', name='en', split=split if split=='train' else 'dev')
        else:
            if task not in self.glue_datasets and task not in self.superglue_datasets:
                dataset = load_dataset(task, split=split)
            else:
                benchmark = 'glue' if task not in self.superglue_datasets else 'super_glue'
                dataset = load_dataset(benchmark,
                                       task.replace('_superglue', '').replace('_bool', ''),
                                       split=split)

        # filtering out empty rows for yahoo
        if self.task == "yahoo_answers_topics":
        # for yahoo dataset we need to filter out empty rows (no question)
            if split=='train':
                good_id = np.load('good_id_yahoo_train.npy')
                dataset = dataset.select(good_id)
            elif split=='test':
                good_id = np.load('good_id_yahoo_test.npy')
                dataset = dataset.select(good_id)

        if self.task == 'wsc': # using only positive samples (for wsc generation)
            idx = np.where(np.array(dataset['label']) == 1)[0]
            dataset = dataset.select(idx)

        if k!=-1:
            dataset = self.select_subset_ds(dataset, k=k)

        if k==-1 and split!='train' and self.task=='multirc':
            # we do not shuffle full validation set of multirc
            # but we save idx of the same questions
            self.save_multirc_questions_idx(dataset)
        else:
            dataset = dataset.shuffle(seed=seed)

        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len,
                                                                            prefix_list=prefix_list),
                                          batched=False)
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, batch_size=batch_size)

            return dataloader

        else:
            N = len(dataset)
            dataset_val = dataset.select(np.arange(0, N//2))
            dataset_test = dataset.select(np.arange(N//2, N))

            dataloaders_val_test = []
            for dataset in [dataset_val, dataset_test]:
                encoded_dataset = dataset.map(lambda x: self.preprocess_function(x, task,
                                                                                 max_length=max_length,
                                                                                 max_length_target=target_len,
                                                                                 prefix_list=prefix_list),
                                              batched=False)
                encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                                  'target_ids', 'target_mask'])
                dataloader = DataLoader(encoded_dataset, batch_size=batch_size)
                dataloaders_val_test.append(dataloader)

            return dataloaders_val_test


class TemplateDatasetForClassification:#(T5Dataset):
    def __init__(self,tokenizer, task, padding='max_length'):
        # super(TemplateDatasetForClassification,self).__init__(tokenizer, task, padding='max_length')
        self.padding = padding
        self.tokenizer = tokenizer

        def get_task_dict(task):
            
            def pattern_match(patterns, source_list):
                
                
                task_names = set()
                for pattern in patterns:
                    for matching in fnmatch.filter(source_list, pattern):
                        task_names.add(matching)

                return list(task_names)
      
            task_names = pattern_match(task.split(","), tasks.ALL_TASKS)
            
            if len(task_names) == 1:
            
                task = lm_eval.tasks.get_task_dict(task_names)
            else:
                return None

            return list(task.values())[0]

        self.task = get_task_dict(task)

      
      

    def get_template(self,):
        dn = self.task.DATASET_PATH

        if self.task.DATASET_NAME is not None:
            dn = f"{dn}/{self.task.DATASET_NAME}"
        if self.task.DATASET_PATH == 'stas/wmt16-en-ro-pre-processed':
            dn = "enro"
        # dn = f"{dn}/{self.task.DATASET_NAME}"
        templates = DatasetTemplates(dn)
        return list(templates.templates.values())
    
    def training_set(self):

        if self.task.has_training_docs():
            return self.task.training_docs()

    def validation_set(self):

        if self.task.has_validation_docs():
            return self.task.validation_docs()

    def testing_set(self):
        if self.task.has_test_docs():
            return self.task.test_docs()

    def apply_template(self,dataset,templates):
        task_set = {'text':[],'label':[]}
        
        for point in dataset:
            for template in templates:
                prompt = template.apply(point)
                label = template.get_answer_choices_list(point)
                task_set['text'].append(prompt[0])
                task_set['label'].append(prompt[1])

        return task_set
    
    def get_final_ds(self, 
        task, 
        split,
        batch_size,
        k=1,
        seed=42,
        return_test=False,
        target_len=2,
        max_length=512,
        prefix_list=[],
        train_amount=1000,
        **kwargs,
        ):

        cut = kwargs['cut']
        task_templates = self.get_template()
        print(task)
 
        np.random.seed(seed)
        
        shuffle = False

    
        # dataset
        if split=='train' and self.training_set() is not None:
            dataset = self.training_set()
            dataset =  dataset[:k]
        else: #'val' in split and self.validation_set() is not None:
            dataset = list(self.validation_set())[:k]
        # if split=='test' and self.testing_set() is not None:
        #     dataset = list(self.testing_set())


        if shuffle:
            np.random.shuffle(dataset)
        
        if task == 'dart':
            processed_dart = []
        
            for d in dataset:
                text = process_triples(d['tripleset'])
                label = d['annotations']['text'][0]
                processed_dart.append({'text':text,'label':label})

        
            dataset = processed_dart

        templates = []

        for i in range(len(task_templates)):
            if split == 'train':
                if i< cut or i >= cut+2:
                    templates.append(task_templates[i])
            else:
                if i>= cut and i < cut+2:
                    templates.append(task_templates[i])
  
        dSet = self.apply_template(dataset,templates)
        
        dataset = hf_Dataset.from_dict(dSet)
        batch_size = batch_size
        
       
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len),batched=False)
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, shuffle=True, batch_size=batch_size)
        
        return dataloader


    def preprocess_function(self, examples, task='some',
                            max_length=512, max_length_target=2,
                            prefix_list=[]):
        tokenizer = self.tokenizer
        if 'text' in examples:
            text = examples['text']
            target = examples['label']
        elif 'article' in examples:
            text = examples['article']
            target = examples['highlights']
        elif 'en' in examples:
            text = examples['en']
            target = examples['ro']
        
       

        source = tokenizer(text,
                            truncation=True,
                            padding=self.padding,
                            #padding='max_length',
                            max_length=max_length)

        target = tokenizer(
                    target,
                    max_length=max_length_target,
                    #pad_to_max_length=True,
                    pad_to_max_length=self.padding=='max_length',
                    #return_tensors="pt"
                    )

        dict_final = {"source_ids": source['input_ids'],
                      "source_mask": source['attention_mask'],
                      "target_ids": target['input_ids'],
                      "target_mask": target['attention_mask']}
        return dict_final

    def get_multi_final_ds(self, 
        task, 
        split,
        batch_size,
        k=1,
        seed=42,
        return_test=False,
        target_len=2,
        max_length=512,
        prefix_list=[],
        train_amount=1000,
        **kwargs,
        ):

        task_templates = self.get_template()
        
        np.random.seed(seed)
        
        shuffle = True

    
        # dataset
        if split=='train' and self.training_set() is not None:
            dataset = self.training_set()
            dataset =  dataset[:k]
        else: #'val' in split and self.validation_set() is not None:
            dataset = list(self.validation_set())[:k]
        # if split=='test' and self.testing_set() is not None:
        #     dataset = list(self.testing_set())


        if shuffle:
            np.random.shuffle(dataset)
        
        if task == 'dart':
            processed_dart = []
        
            for d in dataset:
                text = process_triples(d['tripleset'])
                label = d['annotations']['text'][0]
                processed_dart.append({'text':text,'label':label})

        
            dataset = processed_dart

        templates = task_templates

        dSet = self.apply_template(dataset,templates)
        
        dataset = hf_Dataset.from_dict(dSet)
        batch_size = batch_size
        
       
        if return_test==False:
            encoded_dataset = dataset.map(lambda x: self.preprocess_function(x,
                                                                            max_length=max_length,
                                                                            max_length_target=target_len),batched=False)
            encoded_dataset.set_format(type='torch', columns=['source_ids', 'source_mask',
                                                              'target_ids', 'target_mask'])
            dataloader = DataLoader(encoded_dataset, shuffle=True, batch_size=batch_size)
        
        return dataloader

