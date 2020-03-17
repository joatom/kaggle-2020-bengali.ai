# kaggle-2020-bengali.ai

This repository contains some of my codings for the 2020 kaggle Bengali.AI competition (https://www.kaggle.com/c/bengaliai-cv19).

My submission scored **[78st Place](https://www.kaggle.com/c/bengaliai-cv19/leaderboard) out of 1955** which results in silver medal.

The challange was to classify three components of bengalie handwritten letters. Bengalie letters are build of three components (Grapheme Root, Vowel Diacritics, Consonant Diacritics), which are combined in several ways to build the actual letter. Graphem Roots contains 168 classes, Vowel Diacritics 11 classes and Consonant Diacritics 7 classes.
The train set contains 200840. The hidden test set is about the same size. The hidden test set contains letters that are not present in the train set, but can be build combining components of the train set.

## Models
This solution consists of five models which are ensembled using majority vote. All models contain three heads (one for each letter component) which slightly defer from eachother. The models are build with the fastai library.
- Model 1: pretrained and modified resnext50_32x4d (from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 2: pretrained and modified se_resnext50_32x4d (from [cadene pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 3: pretrained and modified resnext101_32x8d (from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 4: pretrained and modified efficientnet-b2 (from [lukemelas EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)), trained for 120 epochs + 3 epochs (fit_one_cycle)
- Model 5: pretrained and modified efficientnet-b4 (from [lukemelas EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)), trained for 2 x 10 epochs (fit_one_cycle)
 

## Augmentation


## Overview of notebooks

