#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from tqdm import tqdm
import torch
import transformers
from torch.utils.data import Dataset
from datasets import Dataset as Dataset_HF
from transformers import Trainer
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from transformers import pipeline

import pandas as pd
import numpy as np

import os
#import utils
import datasets
import io
import json

#################################
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

#################################


os.environ['TRANSFORMERS_CACHE'] = './cache'

os.environ["WANDB_DISABLED"] = "true"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<PAD>"  #"[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_INST_BOS_TOKEN = "[INST]"
DEFAULT_INST_EOS_TOKEN = "[/INST]"
DEFAULT_SYS_BOS_TOKEN = "<<SYS>>"
DEFAULT_SYS_EOS_TOKEN = "<</SYS>>"


special_dict = {}
#special_dict = {'additional_special_tokens': [DEFAULT_INST_BOS_TOKEN, DEFAULT_INST_EOS_TOKEN, DEFAULT_SYS_BOS_TOKEN, DEFAULT_SYS_EOS_TOKEN]}


PROMPT_DICT = {
    "prompt_no_input": (
        "Rewrite the following toxic input into non-toxic version. Let's break the input down step by step to rewrite the non-toxic version. You should first think about the expanation of why the input text is toxic. Then generate the detoxic output. You must preserve the original meaning as much as possible.\n Input: {toxic}\n"
    ),
  "prompt_pipeline": "Rewrite the following toxic input into non-toxic version. Let's break the input down step by step to rewrite the non-toxic version. You should first think about the expanation of why the input text is toxic. Then generate the detoxic output. You must preserve the original meaning as much as possible.\n Input: "
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "Path to the data directory."})
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    predict_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    debugging: Optional[bool] = field(default=False, metadata={"help": "debugging mode."})
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    predict_with_generate: Optional[bool] = field(default=True)
    source_max_length: Optional[int] = field(default=256)
    target_max_length: Optional[int] = field(default=256)
    

    
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length= training_args.source_max_length + training_args.target_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, debugging: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # import pdb; pdb.set_trace()
        
        if data_path.endswith("json"):
            list_data_dict = jload(data_path)
        
        else:
            if data_path.endswith("csv"):
                df = pd.read_csv(data_path)
            elif data_path.endswith("tsv"):
                df = pd.read_csv(data_path, delimiter='\t')
            elif data_path.endswith("xlsx"):
                df = pd.read_excel(data_path)

            #df = df.head(500000) ############################
            for col in df.columns:
                df[col] = df[col].astype(str)
            list_data_dict = Dataset_HF.from_dict(df)
            # # MBZUI instrutions -> use full100% for training
            # list_data_dict = load_dataset("mbzuai-distil/instruction", split="train", cache_dir="./cache")
            # list_data_dict = list_data_dict.rename_column("response", "output")
            # list_data_dict = list_data_dict.add_column("input", ['']*len(list_data_dict))
            
            #list_data_dict.save_to_disk("instr_train.hf")
            # select few examples for debuggingging
            if debugging:
                print("debugging mode: using only 1000 examples")
                list_data_dict = list_data_dict.select(range(1000))
            
            
        
        logging.warning("Formatting inputs...")
        prompt_no_input = PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_no_input.format_map(example)
            for example in tqdm(list_data_dict)
        ]
        print(sources[0])
        targets = [f"Explanation: {example['explanation']}\nDetoxification: {example['non_toxic']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path, debugging=data_args.debugging)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path, debugging=data_args.debugging)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def main():

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        #cache_dir=training_args.cache_dir,
    )
    model.config.use_cache=False

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        #cache_dir=training_args.cache_dir,
        #model_max_length=training_args.model_max_length,
        #padding_side="right",
        #use_fast=False,
    )
    #if tokenizer.pad_token is None:
    if tokenizer.pad_token is None:
        special_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_dict["unk_token"] = DEFAULT_UNK_TOKEN
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_dict,
        tokenizer=tokenizer,
        model=model,
        )
    '''if "llama" in str(model_args.model_name_or_path).lower():
        tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
                #"eos_token": DEFAULT_EOS_TOKEN,
                #"bos_token": DEFAULT_BOS_TOKEN,
                #"unk_token": DEFAULT_UNK_TOKEN,
            }
        )'''

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    #print(predict_dataset[0])
    
    # update training args to make output dir
    output_dir = os.path.join(training_args.output_dir, model_args.model_name_or_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    
    training_args.output_dir = output_dir
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    # resume from last checkpoint if it exists     
    checkpoint = get_last_checkpoint(training_args.output_dir)

    '''if checkpoint:
        print(f"Checkpoint found! Training from {checkpoint} checkpoint!")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:'''
    print(f"No checkpoint found! Training from scratch!")
    trainer.train()
    
    trainer.save_model()  # Saves the tokenizer too for easy upload
    
    # save states 
    trainer.save_state()
    '''safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    print(f"Training finished! Saved model to {training_args.output_dir}.")'''
    
    '''predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
    metrics = predict_results.metrics
    #metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)'''
    
    print('##### PREDICTION  #####')
    
    
    predict_dataset = pd.read_csv(data_args.predict_data_path)
    src_input = predict_dataset['toxic'].tolist()
    references = predict_dataset['non_toxic'].tolist()
    explanations = predict_dataset['explanation'].tolist()
    
    model_input = []
    for src in src_input:
      model_input.append(str(src)+'\n')
      
    generator = pipeline('text-generation', model=training_args.output_dir, tokenizer = training_args.output_dir, device=0)
    generated_text = generator(model_input, prefix=PROMPT_DICT["prompt_pipeline"], return_full_text=False, clean_up_tokenization_spaces=True, max_new_tokens=512, )
    
    predictions = []
    
    #print(generated_text)
    
    for gen in generated_text:
      predictions.append(gen[0]['generated_text'])
      
    predictions = [pred.strip() for pred in predictions]
    references = [ref.strip() for ref in references]
    sources = [src.strip() for src in src_input]
    
    output_prediction_file = os.path.join(training_args.output_dir, "Xplatform_generated_predictions.jsonl")

    with open(output_prediction_file, "w") as writer:             
        for src, ref, exp, pred in zip(sources, references, explanations, predictions):
            output = {
                'source': src,
                'reference': ref,
                'explnation': exp,
                'prediction': pred,
            }
            json.dump(output, writer)
            #writer.write("REFERENCES: "+str(ref))
            #writer.write("\nPREDICTION: "+str(pred))
            writer.write("\n")
    
    '''
    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            #predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            print(predictions.shape)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]

            references = []
            for ind in range(len(predict_dataset)):
              references.append(predict_dataset[ind]['labels'])
              
            references = np.where(references != -100, references, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(
                references, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            references = [ref.strip() for ref in references]

            
            src_input = []
            for ind in range(len(predict_dataset)):
              src_input.append(predict_dataset[ind]['input_ids'])
              
            #src_input = predict_dataset['input_ids']
            src_input = np.where(src_input != -100, src_input, tokenizer.pad_token_id)
            src_input = tokenizer.batch_decode(
                src_input, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            sources = [src.strip() for src in src_input]

            output_prediction_file = os.path.join(training_args.output_dir, "Xplatform_generated_predictions.jsonl")

            with open(output_prediction_file, "w") as writer:             
                for src, ref, pred in zip(sources, references, predictions):
                    output = {
                        'source': src,
                        'reference': ref,
                        'prediction': pred,
                    }
                    json.dump(output, writer)
                    #writer.write("REFERENCES: "+str(ref))
                    #writer.write("\nPREDICTION: "+str(pred))
                    writer.write("\n")
            '''
    
    


if __name__ == "__main__":
    main()
