# MSA_Medical

## Installation

1. Clone the QueryWise repo. 
```
git clone https://github.com/rajankita/QueryWise.git
cd QueryWise
```

2. Our code is tested on Python 3.8 and Pytorch 1.11.0. Please install the environment via
```
pip install -e requirements.txt
```

3. To run the QueryWise project, you need to install an external model RadFormer. 
```
git clone https://github.com/authorname/RadFormer.git
cd RadFormer
```
4. Modify local paths in RadFormer
- In `RadFormer/models/resnet.py` set `LOCAL_WEIGHT_PATH = "./victim_models/init_resnet/resnet50.pth"`.
- In `RadFormer/models/bagnet.py` modify lines 18-22 to 
```
model_local_paths = {
        "BagNet33": "./victim_models/init_bagnet/bagnet33.pth",
        "BagNet17": "./victim_models/init_bagnet/bagnet17.pth",
        "BagNet9": "./victim_models/init_bagnet/bagnet9.pth",
    }
``` 


We provide support for stealing medical imaging models for two use-cases: Gall Bladder Cancer (GBC) identification, and COVID-19 classification.

## Download Model weights
Download the following victim models.
1. **Radformer** - Download model weights from the official [RadFormer](https://github.com/sbasu276/RadFormer) repo ([this link](https://drive.google.com/file/d/151pPVWQBR5M3RdZW4a616y9VVHl0uZBc/view)). Unzip the model weights and keep them in the `victim_models` directory. 
2. **POCUS-ResNet18** - Download model weights from this link. (Upload weights and provide link here), and keep it in `victim_models` directory. 

In addition, download ImageNet-pretrained model weights from [here](upload and provide link) and keep in `ckpts` directory.

## Prepare Datasets
1. **Gall Bladder Cancer Ultrasound (GBCU)** - This is the GBC victim dataset. To obtain the dataset, follow the instructions [here](https://gbc-iitd.github.io/data/gbcu). Keep the dataset in the `data_msa_medical/GBCU-Shared` directory. 
2. **Gall Bladder Ultrasound Video (GBUSV)** - This is the GBC thief dataset. To get the dataset, follow the instructions [here](https://gbc-iitd.github.io/data/gbusv). Keep the dataset in the `data_msa_medical/GBUSV-Shared` directory.  
3. **POCUS** - This is the COVID-19 victim dataset. Follow the instructions on [USCL repo](https://github.com/983632847/USCL) to get the 5 fold cross-validation POCUS dataset. Keep it in `data_msa_medical/covid_5_fold`.
4. **COVIDx-US** - THis is the COVID-19 thief dataset. Follow the instructions on the [COVIDx-US repo](https://github.com/nrc-cnrc/COVID-US) to generate the dataset, and keep it in `data_msa_medical/covidx_us`.


## Run model stealing baselines
Use the script `activethief/train.py` to train a thief model. Config files are named as `<victim_arch>_<thief_arch>_<thief_dataset>.yaml`. 

```
cd activethief
python activethief/train.py --c activethief/configs/gbc/radformer_resnet50_gbusv.yaml
```
The default values in the config files support Random sample selection from KnockoffNets. To run k-Center or Entropy, edit the config to change `ACTIVE.METHOD` to 'kcenter' or 'entropy' respectively, and `ACTIVE.CYCLES` to 5.

## Run proposed method
Note that for the proposed method, you must first run the baseline method to train the anchor model. 
Use the script `train_proposed.py` to train a thief model using SSL. Edit the config file to set the appropriate data paths, and path to the anchor model as well as the labeled set queried from the victim during training the anchor model.

```
cd ssl
python train_proposed.py --c configs/gbc/querywise_resnet50.yaml
```

## Acknowledgements
Parts of this codebase are built upon
- [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning)
- [tribhuvanesh/knockoffnets](https://github.com/tribhuvanesh/knockoffnets)

Thanks to the authors of these papers for making their code available for public usage. 