import pandas as pd
from fastai.vision import *


# load labels (train target)
def load_labels(source = Path('/home/kaggle/bengaliai-cv19/input')):
    df_label = pd.read_csv(source/'train.csv')
    return df_label.drop('grapheme', axis=1)


# reads all images
def read_all(source = Path('/home/kaggle/bengaliai-cv19/input'), file='train_image_data_crop_scaled_'):
    df = (pd.read_feather(source/(file+'0.feather'))
          .append(pd.read_feather(source/(file+'1.feather')))
          .append(pd.read_feather(source/(file+'2.feather')))
          .append(pd.read_feather(source/(file+'3.feather')))
         )
    df.reset_index(inplace=True,drop = True)
    return df


# combine labels and images
def label_images():
    df_label = load_labels()
    df_train = read_all()
    df_train = df_train.merge(df_label, on='image_id')
    df_train['fn'] = df_train.index
    return df_train

    
# modified version of https://www.kaggle.com/melissarajaram/model-ensembling-and-transfer-learning
class PixelImageItemList(ImageList):
    
    def open(self,fn):
        SIZE = 128
        img_pixel = self.inner_df.loc[self.inner_df['fn'] == int(fn[2:])].values[0,1:(SIZE*SIZE+1)]
        img_pixel = img_pixel.reshape(SIZE,SIZE)
        return vision.Image((pil2tensor(img_pixel,np.float32).div_(255)-1).abs_())    

    
# create databunch    
def data(bs):
    
    df_train = label_images()
    piil = PixelImageItemList.from_df(df=df_train,path='.',cols='fn')
    tfms = get_transforms(do_flip=False,max_zoom=1, xtra_tfms=[cutout(n_holes=(1,4), length=(16, 16), p=.5)]) 

    return (piil
        .split_by_rand_pct(0.2)
        .label_from_df(cols=['grapheme_root','vowel_diacritic','consonant_diacritic'])
        .transform(tfms, padding_mode='zeros')
        .databunch(bs=bs).normalize(imagenet_stats)) 