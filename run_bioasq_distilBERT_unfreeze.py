#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.
import tensorflow as tf
import torch
import argparse
import glob
import logging
import os
import random
import timeit
import json
import shutil
import re
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset,DataLoader, RandomSampler, SequentialSampler
from itertools import product,permutations,combinations,combinations_with_replacement
import os.path,subprocess
from subprocess import call,Popen,PIPE
from shutil import copyfile
import time


from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup

)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


args = {
    'data_dir':'/BioBERT/BioBERT-PyTorch/data_dir/',
    'model_type':  'bert',
    'model_name': '/BioBERT/BioBERT-PyTorch/BioBERTv1.1-SQuADv1.1-Factoid-PyTorch/',
    'output_dir': '/BioBERT/BioBERT-PyTorch/outputs_default_3/',
    'do_train': True,
    'do_eval': True,
    'per_gpu_train_batch_size':3,
    'per_gpu_eval_batch_size':8,
    'num_train_epochs': 2,
    'max_seq_length': 512,
    'train_batch_size': 3,
    'eval_batch_size': 3,
    'do_lower_case' : False,
    'train_file':'BioASQ_trainingdata_9b.json',
    'predict_file':'BioASQ-testset5.json',
    'gradient_accumulation_steps': 1,
    'weight_decay': 0,
    'learning_rate': 3e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'doc_stride':128,
    'max_query_length':64,
    

    'null_score_diff_threshold':0,
    'local_rank':-1,
    'fp16': False,
    'fp16_opt_level': '01',
    'max_steps':-1,
    'n_best_size':20,
    'max_answer_length':30,
    'verbose_logging': True,
    'evaluate_during_training':False,
    'version_2_with_negative':False,
    'logging_steps': 900000,
    'evaluate_during_training': False,
    'save_steps': 900000,
    'eval_all_checkpoints': True,
    'seed':42,
    'overwrite_output_dir': False,
    'reprocess_input_data': False,
    'notes': 'BioASQ Dataset'
    
}

def initialize(args1):
  # Set the seed value all over the place to make this reproducible.
  args['output_dir']=args1.output_dir
  seed_val = args1.seed
  random.seed(seed_val)
  filename=args1.args_filename
  args['per_gpu_train_batch_size']=args1.per_gpu_train_batch_size
  args['model_type']=args1.model_type
  args['model_name']=args1.model_name
  args['learning_rate']=args1.learning_rate

  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)
  ###Function to copy the argumants in a json file and delete output_dir folder if it exists
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  args['n_gpu']=torch.cuda.device_count()
  #print(args['n_gpu'])
  os.chdir("/BioBERT/BioBERT-PyTorch")
  time.sleep(10)
  if(os.path.exists(args1.prediction_dir) and os.listdir(args1.prediction_dir)): 
      shutil.rmtree(args1.prediction_dir,ignore_errors = False)
      time.sleep(10)
      print("folder deleted")

  with open(filename, 'w') as f:
    json.dump(args, f)
    if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
      print("Output directory already exists and is not empty")
      shutil.rmtree(args['output_dir'])
      print("removed output dir ")
    
  
   


  
  MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),}
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
  config = config_class.from_pretrained(args['model_name'])
  tokenizer = tokenizer_class.from_pretrained(args['model_name'],do_lower_case=args['do_lower_case'])
  model = model_class.from_pretrained(args['model_name'])
  model.to(device);
  #device
  #Setup logging
  logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args['local_rank'] in [-1, 0] else logging.WARN,)
  logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args['local_rank'],device,args['n_gpu'],bool(args['local_rank'] != -1),args['fp16'])
  return(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
  
  
### Function to load training and test dataset,convert into SQuAD format( first into examples and then features) for different models
### Based on run_squad.py from https://github.com/huggingface/transformers


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_examples(args, tokenizer, evaluate=False, output_examples=False):
  input_dir = args['data_dir']
  processor = SquadV1Processor()
  if evaluate:
    examples = processor.get_dev_examples(args['data_dir'], filename=args['predict_file'])
    logger.info("Creating features from test dataset file at %s", input_dir)
  else:
    examples = processor.get_train_examples(args['data_dir'], filename=args['train_file'])
    logger.info("Creating features from training dataset file at %s", input_dir)
    
  features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args['max_seq_length'],
            doc_stride=args['doc_stride'],
            max_query_length=args['max_query_length'],
            is_training=not evaluate,
            return_dataset="pt" )
  
  if output_examples:
    return dataset, examples, features
  return dataset


def print_optimizer_parameters(optimizer):
  format_string =  ' ('
  for i, group in enumerate(optimizer.param_groups):
    format_string += '\n'
    format_string += 'Parameter Group {0}\n'.format(i)
    for key in sorted(group.keys()):
      if key != 'params':
        format_string += '    {0}: {1}\n'.format(key, group[key]) 
  format_string += ')'
  return print(format_string)


def freeze_layers(model,n):
  for param in list(model.distilbert.embeddings.parameters()):
    param.requires_grad = False
  print ("Embedding Layer Frozen")
  layers_freeze=[]
  for i in range(n):
    layers_freeze.append(i)
  for layer_idx in layers_freeze:
    for param in list(model.distilbert.transformer.layer[layer_idx].parameters()):
      param.requires_grad = False
    print("Layer {} frozen: " .format(layer_idx))

def unfreeze_layers_withembedding(model,n):
    for param in list(model.distilbert.embeddings.parameters()):
           param.requires_grad=True
    print ("Embedding Layer Frozen")
    n=abs(n)
    n-=1
    layers_freeze=[]
    for i in range(5,n,-1):
        layers_freeze.append(i)
    for layer_idx in layers_freeze:
        for param in list(model.distilbert.transformer.layer[layer_idx].parameters()):
            param.requires_grad = True
        print("Layer {} frozen: " .format(layer_idx))


def unfreeze_layers(model,n):
  n=abs(n)
  n-=1
  layers_freeze=[]
  for i in range(5,n,-1):
    layers_freeze.append(i)
  for layer_idx in layers_freeze:
    for param in list(model.distilbert.transformer.layer[layer_idx].parameters()):
      param.requires_grad = True
    print("Layer {} frozen: " .format(layer_idx))

  
    

def print_summary(model):
  params_bert = list(model.named_parameters())
  count=0
  for n,p in model.named_parameters():
    if p.requires_grad:
      count+=1
  print('The BERT model has in total {:} different named parameters out of which {:} are trainable \n'.format(len(params_bert),count))  
  
  header = ["Layer name", "Output Shape", "Trainable"]
  print("============================================================\n")
  print(f"{header[0]:<30} {header[1]:<20} {header[2]:<10} \n")
  print("============================================================\n")
  
  for n,p in model.named_parameters():
    if 'embedding' in n:
      print(n,'\t\t\t', p.shape, '\t',p.requires_grad)
    elif 'distilbert'in n:
      print(n,'\t\t\t', p.shape, '\t',p.requires_grad)
    else:
      print(n,'\t\t\t', p.shape, '\t',p.requires_grad)
  print('\n')
  total_params=total_no_params(model)
  trainable_params=total_trainable_params(model)
  non_trainable_params=total_params-trainable_params
  print("Total number of parameters:{}".format(total_params))
  print("Total number of trainable parameters:{}".format(trainable_params))
  print("Total number of non-trainable parameters:{}".format(non_trainable_params))
  print(f"We will start by training {trainable_params:3e} parameters out of {total_params:3e},"f" i.e. {100 * trainable_params/total_params:.2f}%") 


def unfreeze_all(model):
  for param in list(model.distilbert.embeddings.parameters()):
    param.requires_grad = True
  print ("Embedding Layer unfrozen")
  for param in list(model.distilbert.transformer.parameters()):
    param.requires_grad = True
  print("All layers unfrozen")

def freeze_all(model):
  for param in list(model.distilbert.embeddings.parameters()):
    param.requires_grad = False
  print ("Embedding Layer Frozen")
  for param in list(model.distilbert.transformer.parameters()):
    param.requires_grad = False
  print("All layers frozen")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

#Function to evaluate the model
### Based on run_squad.py from https://github.com/huggingface/transformers

def evaluate(args, model, tokenizer,device, prefix=""):
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            
            if args['model_type'] in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args['output_dir'], "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args['output_dir'], "nbest_predictions_{}.json".format(prefix))
    if args['version_2_with_negative']:
        output_null_log_odds_file = os.path.join(args['output_dir'], "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None
    predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args['n_best_size'],
            args['max_answer_length'],
            args['do_lower_case'],
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args['verbose_logging'],
            args['version_2_with_negative'],
            args['null_score_diff_threshold'],
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results
	
	
def total_no_params(model):
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  return(pytorch_total_params)
#If you want to calculate only the trainable parameters:
def total_trainable_params(model):
  pytorch_total_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return(pytorch_total_params1)
  
def train(args, train_dataset, model, tokenizer,device,num_train_epochs):
    
    """ Train the model """
    #if args['local_rank'] in [-1, 0]:
        #tb_writer = SummaryWriter()
    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    #set num_train_epochs for gradual unfreezing  
    args['num_train_epochs'] =num_train_epochs
    

    # Total number of training steps is number of batches * number of epochs.
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight'] 

    optimizer_grouped_parameters = [
        {'params': [p for n, p in filter(lambda n: n[1].requires_grad,model.named_parameters()) if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in filter(lambda n: n[1].requires_grad,model.named_parameters()) if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args['warmup_steps'], num_training_steps = t_total)



    # Train!

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    training_loss=[] #used for plotting graphs
    learning_rate_p=[]
    testing_loss=[]
    results_plotting = {} 
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    print(train_iterator)
    for i in train_iterator:
        print("")
        print('======== Epoch {:} / {:} ========'.format( i+1, args['num_train_epochs']))
        print('Training...')
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}


            if args['model_type'] in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
                del inputs["token_type_ids"]


            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #print("\r%f" % loss, end='')
            
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                     scaled_loss.backward()
            else:
                loss.backward()
                

            tr_loss += loss.item()
            
            
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])
              
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
				
				  
				        # Log metrics
                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args['local_rank'] == -1 and args['evaluate_during_training']:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            #tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info("eval_{} value %s".format(key),value)
                        logger.info("global step %s",global_step)
                        logger.info("Results exact= %s, f1= %s after %s epoch at global step %s",results['best_exact'],results['best_f1'],i+1,global_step)
                        results = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in results.items())
                        results_plotting.update(results)
                        logger.info("Results: {}".format(results_plotting))
                        logger.info("logging steps %s",args['logging_steps'])
                    #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    logger.info("learning rate =%s at global step = %s",scheduler.get_lr()[0], global_step)
                    #tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args['logging_steps'], global_step)
                    logger.info("loss %s, global_step %s",(tr_loss-logging_loss)/args['logging_steps'],global_step)
                    logging_loss = tr_loss
                    
                    

                    
                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
        
        logger.info("Sum of training loss at the end of Epoch %s = %s ", i+1,tr_loss)
        logger.info("Average training loss at the end of Epoch %s = %s ", i+1,tr_loss/global_step)
        # Calculate the average loss over the training data.
        #avg_train_loss = tr_loss / global_step            
    
        # Store the loss value for plotting the learning curve.
        #training_loss.append(avg_train_loss)
        
    #if args['local_rank'] in [-1, 0]:
      #tb_writer.close()
    
    return global_step, tr_loss,tr_loss / global_step
	
	
def evaluation_squad(args,model,tokenizer,model_class,device):
  results = {}
  if args['do_eval']:
        if args['do_train']:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args['output_dir']]
            if args['eval_all_checkpoints']:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args['output_dir'] + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args['model_name_or_path'])
            checkpoints = [args['model_name_or_path']]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(device)

            # Evaluate
            result = evaluate(args, model, tokenizer, device,prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)
            
  logger.info("Results: {}".format(results))
  return(results)

def evaluation_bioasq(args1):
    os.chdir("/BioBERT/BioBERT-PyTorch")
    print(args1.prediction_dir)
    print(args1.output_dir)
    path=args1.prediction_dir
    print(path)
    fullpath="/BioBERT/BioBERT-PyTorch"
    os.makedirs(path)
    path_from=os.path.join(args1.output_dir,"nbest_predictions_.json")
    path_to1=os.path.join(args1.prediction_dir,"nbest_predictions_.json")
    path_to=args1.prediction_dir
    print("path_to",path_to)
    print("path_from",path_from)
    shutil.copy(path_from, path_to)
    cmd1="python3 /BioBERT/BioBERT-PyTorch/transform_n2b_factoid.py --nbest_path {}  --output_path  {}".format(path_to1,args1.prediction_dir)
    os.system(cmd1)
    path2_to=os.path.join(args1.prediction_dir,"9B5_golden.json")
    print("path2_to",path2_to)
    shutil.copy("/BioBERT/BioBERT-PyTorch/9B5_golden.json", path_to)
    path3_to=os.path.join(args1.prediction_dir,"BioASQform_BioASQ-answer.json")
    print("path3_to",path3_to)
    os.chdir("/BioBERT/BioBERT-Evaluation-Measures")
    cmd="java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 {}  {}".format(path2_to,path3_to)
    proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE,close_fds=False)
    proc.wait()
    (stdout, stderr) = proc.communicate()
    if proc.returncode != 0:
       print(stderr)
    result=stdout.decode("utf-8")
    print(result)
    time.sleep(10)
    #proc.stdout.flush()
    proc.kill()
    #kill(proc.pid)
    print(result)
    li=list(map(float, result.strip().split()))
    listOfStr = ["0", "strict_accuracy" , "lenient_accuracy" , "mean_reciprocal_rank" , "1" , "2","3","4","5","6" ]
    zipbObj = zip(listOfStr, li)
    dictOfWords = dict(zipbObj)
    print(dictOfWords)
    return(dictOfWords)








# Save the trained model and the tokenizer
def save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device):
  if args['do_train']:
        # Create output directory if needed
        if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
        logger.info("Saving model checkpoint to %s", args['output_dir'])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args['output_dir'])
        tokenizer.save_pretrained(args['output_dir'])

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args['output_dir'], "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        #print(config_class,model_class,tokenizer_class,args['model_type'])
        model = model_class.from_pretrained(args['output_dir'])  # , force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args['output_dir'], do_lower_case=args['do_lower_case'])
        model.to(device)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_type", default=None,type=str, help="Model type selected in the list")
    parser.add_argument("--model_name",default=None,type=str,help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--embedding", action="store_true", help="Whether to include embedding layers during gradual unfreezing.")
    parser.add_argument("--unfreeze_tlayer_at_a_time" ,default=None, type=int, help="Number of transformer layers to unfreeze at a time after unfreezing top linear layers.")
    parser.add_argument("--l" ,default=None, type=int, help="Number of transformer layers in the model")
    parser.add_argument("--start", default=None, type=int, help="Starting range for epochs for hypertuning")
    parser.add_argument("--stop", default=None, type=int,  help="Stopping range for epochs for hypertuning")
    parser.add_argument("--increment", default=1, type=int,  help="Value to increment on epochs for hypertuning")
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--prediction_dir", default=None, type=str, required=True,help="The directory where the model predictions are copied to evaluate using BioASQ.")
    parser.add_argument("--args_filename", default=None, type=str, required=True,help="The name of the json file where the arguments of the script wii be written.")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,help="Batch size per GPU/CPU for training.")
    parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
    parser.add_argument("--learning_rate", default=5e-5, type=float,help="The initial learning rate for Adam.")
    args1 = parser.parse_args()
    r=[]
    print(args1)
    results_squad={}
    results_bioasq={}
    for i in range(args1.start,args1.stop+1,args1.increment):
        r.append(i)
    print(r)
    no_of_tlayers=args1.l
    print(no_of_tlayers)
    no_of_loops=int(args1.l/args1.unfreeze_tlayer_at_a_time)
    print(no_of_loops)
    if args1.embedding:
        no_of_loops=no_of_loops+2
    else:
        no_of_loops=no_of_loops+1
    print(no_of_loops)
    print(args1.do_train)
    print(args1.embedding)
    print(args1.unfreeze_tlayer_at_a_time)
    count=0
  
     
    if args1.do_train and args1.embedding and args1.unfreeze_tlayer_at_a_time==1:
        print("entered loop")
        j=[[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j[i])
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-5)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-4)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info(" training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-3)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][3])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s",j[i][3], global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-2)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][4])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][4],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-1)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][5])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][5],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][6])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][6],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers_withembedding(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][7])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][7],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4],j[i][5],j[i][6],j[i][7]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4],j[i][5],j[i][6],j[i][7]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)
	
    if args1.do_train and not args1.embedding and args1.unfreeze_tlayer_at_a_time==1:
        print("entered loop")
        j=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-5)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-4)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-3)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][3])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s",j[i][3], global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-2)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][4])
            logger.info(" training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][4],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-1)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][5])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][5],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][6])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][6],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4],j[i][5],j[i][6]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4],j[i][5],j[i][6]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)
	

   
    
    if args1.do_train and args1.embedding and args1.unfreeze_tlayer_at_a_time==2:
        print("entered loop")
        j=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("After unfreezing top 4 linear layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-4)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("After unfreezing top 4 linear layers and top 11,10,9,8th transformer layers and training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-2)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("After unfreezing top 4 linear layers and top 11,10,9,8,7,6,5,4th transformer layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][3])
            logger.info("After unfreezing top 4 linear layers and top 12 transformer layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s",j[i][3], global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers_withembedding(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][4])
            logger.info("After 4 linear layers and  top 12 transformer layers and embedding layer and training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][4],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3],j[i][4]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)


    if args1.do_train and not args1.embedding and args1.unfreeze_tlayer_at_a_time==2:
        print("entered loop")
        j=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("After unfreezing top 4 linear layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-4)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("After unfreezing top 4 linear layers and top 11,10,9,8th transformer layers and training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-2)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("After unfreezing top 4 linear layers and top 11,10,9,8,7,6,5,4th transformer layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][3])
            logger.info("After unfreezing top 4 linear layers and top 12 transformer layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s",j[i][3], global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)
    
    if args1.do_train and args1.embedding and args1.unfreeze_tlayer_at_a_time==3:
        print("entered loop")
        j=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("After unfreezing top 4 linear layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-3)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers_withembedding(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][3])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}_{}".format(j[i][0],j[i][1],j[i][2],j[i][3]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)
     
        
        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)
 
    if args1.do_train and not args1.embedding and args1.unfreeze_tlayer_at_a_time==3:
        print("entered loop")
        j=[[3,1,1]]
        for i in range(0,1,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,-3)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}".format(j[i][0],j[i][1],j[i][2]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}".format(j[i][0],j[i][1],j[i][2]) if j else ""), v) for k, v in result1.items())
            print("The Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)


    if args1.do_train and args1.embedding and args1.unfreeze_tlayer_at_a_time==6:
        print("entered loop")
        j=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,2]]
        for i in range(0,2,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("After unfreezing top 4 linear layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers_withembedding(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][2])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][2],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}_{}".format(j[i][0],j[i][1],j[i][2]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are %s",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}_{}".format(j[i][0],j[i][1],j[i][2]) if j else ""), v) for k, v in result1.items())
            print("The Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)

        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)

 
    if args1.do_train and not args1.embedding and args1.unfreeze_tlayer_at_a_time==6:
        print("entered loop")
        j=[[5,1]]
        for i in range(0,1,1):
            print("Combination is",j)
            count+=1
            args,tokenizer,config,model,tokenizer_class,config_class,model_class,device=initialize(args1)
            train_dataset = load_examples(args, tokenizer, evaluate=False, output_examples=False)
            freeze_all(model)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][0])
            logger.info("After unfreezing top 4 linear layers and training with %s epochs, the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][0],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            unfreeze_layers(model,0)
            print_summary(model)
            global_step, tr_loss,tr_loss_avg = train(args, train_dataset, model, tokenizer,device,j[i][1])
            logger.info("training with %s epochs,the global_step = %s, tr_loss= %s,total average training loss = %s", j[i][1],global_step, tr_loss,tr_loss_avg)
            save_load_weights(args,tokenizer,config,model,tokenizer_class,config_class,model_class,device)
            result=evaluation_squad(args,model,tokenizer,model_class,device)
            result = dict((k + ("_{}_{}".format(j[i][0],j[i][1]) if j else ""), v) for k, v in result.items())
            print("The Current result is",result)
            logger.info("The current result is %s",result)
            results_squad.update(result)
            print("The results are",results_squad)
            logger.info("The results are %s",results_squad)
            result1=evaluation_bioasq(args1)
            result1 = dict((k + ("_{}_{}".format(j[i][0],j[i][1]) if j else ""), v) for k, v in result1.items())
            print("Current result is",result1)
            logger.info("The current result is %s",result1)
            results_bioasq.update(result1)
            print("The results are",results_bioasq)
            logger.info("The results are %s",results_bioasq)
            
        print("combinations run",count)
        print(results_squad) 
        logger.info("The results are %s",results_squad)
        print(results_bioasq) 
        logger.info("The results are %s",results_bioasq)

 




if __name__ == "__main__":
    main()


