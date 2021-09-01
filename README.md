# bioasq9b
Download our pre-processed version of BioASQ 9b training (BioASQ_trainingdata_9b.json) and Test Batch 5 dataset (BioASQ-testset5.json), and place them in the /BioBERT/BioBERT-PyTorch/data_dir folder. 
Place the DistilBERT weights and the file transform_n2b_factoid.py in the  /BioBERT/BioBERT-PyTorch folder.

Clone the github repository https://github.com/BioASQ/Evaluation-Measures in the /BioBERT folder. 

This code gradually unfreezes 3 transformer layers of DistilBERT at a time. Following command runs fine-tuning code on QA with default arguments. 

python3 run_bioasq_distilBERT_unfreeze.py   
     --i 2 
	 --do_train  
	 --model_type distilbert 
	 --model_name /BioBERT/BioBERT-PyTorch/distilbert-base-cased-distilled-squad/   
	 --per_gpu_train_batch_size 4  
	 --unfreeze_tlayer_at_a_time  3 
	 --l 6
	 --start 1 
	 --stop 3 
	 --output_dir /BioBERT/BioBERT-PyTorch/bioasq_ditilbert_b4     
	 --prediction_dir /BioBERT/BioBERT-PyTorch/prediction_distilbert_b4
	 --args_filename distilbert-b4_3e-5_.json 
	 --learning_rate 3e-5
		 
		 

Citation
Cite the Arxiv paper:

@article{}


or the conference paper:
  
