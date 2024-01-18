# OC-Hunter

## Datesets
The dataset employed by OC-Hunter is a public open source datasets. We can get it from [here](./https://drive.google.com/drive/folders/1FKhZTQzkj-QpTdPE9f_L9Gn_pFP_EdBi)

## Environmant Installation
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
