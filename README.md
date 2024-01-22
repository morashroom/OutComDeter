# OC-Hunter:Co-Matching Attention Mechanisms for Just-In-Time Detection of Obsolete Comments in Software Development

## Introduction
This replication package contains the dataset and code for our paper `OC-Hunter: Co-Matching Attention Mechanisms for Just-In-Time Detection of Obsolete Comments in Software Development`. The work is devoted to efficiently detecting obsolete comments in software development by capturing the complex semantic information between code and comments. The OC-Hunter consists of a Data Processing Component, a Feature Extraction Component, an Attention Component, and a Probability Calculation Component.

<img width="678" alt="16e6f1998cb3b4df21a937631bce5e0" src="https://github.com/morashroom/OC-Hunter/assets/98865514/cbb4600c-f3fa-4d7f-9fa8-e5ae5fe1079e">

## Core Repository Structure
Here are the introduction of the core directories and files in OC-Hunter. 
* configs: This directory contains the train parameters of OC-Hunter like epoch, batch size.
* metrics: This directory contains the related files about evaluation metrics like Precision and Recall.
* models: This directory contains OC-Hunter architecture.
* utils: This directory contains the utility files to process the dataset.
* eval.py: This file is used to evaluate the performance of OC-Hunter. 
* tran.py: This file is used to train OC-Hunter models.
* main.py: This file is used to run the OC-Hunter.

## Dateset Preparation
We can get the dataset from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi). Please download the dataset and follow the below steps to put dataset inside the OC-Hunter folder:
```
mkdir data
mv cup2_dataset.zip data
cd data
unzip cup_dataset.zip
```

## Environment Installation
```
conda env create -f environment.yml
pip install git+https://github.com/Maluuba/nlg-eval.git@81702e
# set the data_path
nlg-eval --setup ${data_path}
sudo apt-get install python2
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
sudo python2 get-pip.py
pip2 install scipy
```
## Run 
```
python -m main run_oc_hunter configs/OC-Hunter.yml OC-Hunter
```
