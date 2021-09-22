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

@inproceedings{a23f5ef4c3ae4df5a7601b0cb2c2b9b7,
title = "Transformer-based language models for factoid question answering at BioASQ9b",
abstract = "In this work, we describe our experiments and participating systems in the BioASQ Task 9b Phase B challenge of biomedical question answering. We have focused on finding the ideal answers and investigated multi-task fine-tuning and gradual unfreezing techniques on transformer-based language models. For factoid questions, our ALBERT-based systems ranked first in test batch 1 and fourth in test batch 2. Our DistilBERT systems outperformed the ALBERT variants in test batches 4 and 5 despite having 81% fewer parameters than ALBERT. However, we observed that gradual unfreezing had no significant impact on the model's accuracy compared to standard fine-tuning.",
keywords = "ALBERT, BioASQ9b, DistilBERT, Question answering, Transfer learning",
author = "Urvashi Khanna and Diego Moll{\'a}",
note = "Publisher Copyright: {\textcopyright} 2021 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).; 2021 Working Notes of CLEF - Conference and Labs of the Evaluation Forum, CLEF-WN 2021 ; Conference date: 21-09-2021 Through 24-09-2021",
year = "2021",
language = "English",
series = "CEUR Workshop Proceedings",
publisher = "CEUR",
pages = "247--257",
editor = "Guglielmo Faggioli and Nicola Ferro and Alexis Joly and Maria Maistro and Florina Piroi",
booktitle = "CLEF 2021 Working Notes",
}
  
