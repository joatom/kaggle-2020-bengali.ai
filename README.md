# kaggle-2020-bengali.ai

This repository contains some of my codings for the 2020 kaggle Bengali.AI competition (https://www.kaggle.com/c/bengaliai-cv19).

My submission scored **[78st Place](https://www.kaggle.com/c/bengaliai-cv19/leaderboard) out of 1955** which results in silver medal.

The challange was to classify three components of bengalie handwritten letters. Bengalie letters are build of three components (Grapheme Root, Vowel Diacritics, Consonant Diacritics), which are combined in several ways to build the actual letter. Graphem Roots contains 168 classes, Vowel Diacritics 11 classes and Consonant Diacritics 7 classes.
The train set contains 200840. The hidden test set is about the same size. The hidden test set contains letters that are not present in the train set, but can be build combining components of the train set.

## Solution Design

### Image preperation
The images are croped, scaled to fill the entire image size and resized (224x224) following this [notebook](https://www.kaggle.com/maxlenormand/cropping-to-character-resizing-images). Another cropping and centering variation without scaling from [here](https://www.kaggle.com/iafoss/image-preprocessing-128x128) didn't work as well for my model. I also run experiments with original images, several image sizes, color inversion and another .   

### Augmentation
Slightly modified default transformation from fastai'
```python
tfms = get_transforms(do_flip=False,max_zoom=1, xtra_tfms=[cutout(n_holes=(1,4), length=(16, 16), p=.5)])
...
.transform(tfms, padding_mode='zeros')
...
...MixUpCallback(learn)...
```
Since my images were scaled I restricted max-zoom to 100%. I randomly used only 1-4 small rectangular cutouts. Padding-Mode was set to "zeros".
Additionaly for augmentation and also as regularization technique I used [this](https://www.kaggle.com/iafoss/grapheme-fast-ai-starter-lb-0-964#MixUp) MixUp implementation, which is an extention of the fastai version for the three head architecture. 

### Validation strategie
All models were trained with a 20% random validation split.
I also tried this StratifiedKFold split which didn't improve my results:
```python
train['tag'] = train['grapheme_root'].astype(str)+'_'+train['vowel_diacritic'].astype(str)+'_'+train['consonant_diacritic'].astype(str)

n_splits=5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)

for fold, (train_idx, val_idx) in enumerate(skf.split(X=train, y=train['tag'].values)):
                    train['valid_kf_'+str(fold)] = 0
                    train.loc[val_idx, 'valid_kf_'+str(fold)] = 1
train.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>grapheme_root</th>
      <th>vowel_diacritic</th>
      <th>consonant_diacritic</th>
      <th>tag</th>
      <th>valid_kf_0</th>
      <th>valid_kf_1</th>
      <th>valid_kf_2</th>
      <th>valid_kf_3</th>
      <th>valid_kf_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Train_0</td>
      <td>15</td>
      <td>9</td>
      <td>5</td>
      <td>15_9_5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Train_1</td>
      <td>159</td>
      <td>0</td>
      <td>0</td>
      <td>159_0_0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Train_2</td>
      <td>22</td>
      <td>3</td>
      <td>5</td>
      <td>22_3_5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Train_3</td>
      <td>53</td>
      <td>2</td>
      <td>2</td>
      <td>53_2_2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Train_4</td>
      <td>71</td>
      <td>9</td>
      <td>5</td>
      <td>71_9_5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


### Models
This solution consists of five models which are ensembled using majority vote. All models contain three heads (one for each letter component) which slightly defer from eachother. The models are build with the fastai library.
- Model 1: pretrained and modified resnext50_32x4d (from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 2: pretrained and modified se_resnext50_32x4d (from [cadene pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 3: pretrained and modified resnext101_32x8d (from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)), trained for 2 x 10 epochs (fit_one_cycle)
- Model 4: pretrained and modified efficientnet-b2 (from [lukemelas EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)), trained for 120 epochs + 3 epochs (fit_one_cycle)
- Model 5: pretrained and modified efficientnet-b4 (from [lukemelas EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)), trained for 2 x 10 epochs (fit_one_cycle)

## Scores



## Overview of notebooks
### Training
- ImagePreprocessing.ipynb: Preparing and saving the images

### Inference

### Additional notes and experiments


## References
- Competition homepage: https://www.kaggle.com/c/bengaliai-cv19
- Image Preprocessing: 
  - https://www.kaggle.com/maxlenormand/cropping-to-character-resizing-images
  - https://www.kaggle.com/iafoss/image-preprocessing-128x128
- Pretrained Models
  - https://pytorch.org/docs/stable/torchvision/models.html
  - https://github.com/Cadene/pretrained-models.pytorch
  - https://github.com/lukemelas/EfficientNet-PyTorch
