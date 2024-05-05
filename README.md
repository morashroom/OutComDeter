# Just-In-Time Detection of Outdated Comments in Software Development by Jointly Reasoning
## Introduction
This replication package contains the dataset and code for our paper `Just-In-Time Detection of Outdated Comments in Software Development by Jointly Reasoning`. The work is devoted to efficiently detecting obsolete comments in software development by capturing the complex semantic information between code and comments. The OutComDeter consists of the data processing component, the jointly reasoning component, the feature extraction component and the outdated comment detection components. Here are the framework of OutComDeter.

<img width="562" alt="a7ba83922735fc9d17c6ea0830ba5b5" src="https://github.com/morashroom/OC-Hunter/assets/98865514/1b97f1c9-474d-45af-94a5-34bd6359635d">


## Core Repository Structure
Here are the introduction of the core directories and files in OutComDeter. 
* configs: This directory contains the train parameters of OutComDeter like epoch, batch size.
* metrics: This directory contains the related files about evaluation metrics like Precision and Recall.
* models: This directory contains OutComDeter architecture.
* utils: This directory contains the utility files to process the dataset.
* eval.py: This file is used to evaluate the performance of OutComDeter. 
* tran.py: This file is used to train OutComDeter models.
* main.py: This file is used to run the OutComDeter.

## Dateset Preparation
We can get the dataset from [here](https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi). Please download the dataset and follow the below steps to put dataset inside the OutComDeter folder:
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
python -m main run_oc_hunter configs/OutComDeter.yml OutComDeter
```
