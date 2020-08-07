import sys
sys.path.append("../..")\

# import recommend
# import common
# from common import FILE_PATH
# from common import MODEL_PATH

FILE_PATH = '/home/youngchang/PycharmProjects/CRM_dir2/rest_server/member/ctr/dataset/'
MODEL_PATH ='/home/youngchang/PycharmProjects/CRM_dir2/rest_server/member/ctr/examples/models/DeepFM.pth'

from member.ctr.examples.shopngine.product_refine import *
from member.ctr.examples.shopngine.recommend import *
from member.ctr.examples.shopngine.common import *

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import pickle
import locale

locale.setlocale(locale.LC_ALL, '')

st = sns.axes_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

plt.rcParams ['font.family'] = 'NanumGothic'

import warnings
warnings.filterwarnings("ignore")

# feather 파일로 저장되어있는 각각의 dataframe을 읽어옴

reviews = pd.read_feather (FILE_PATH + 'reviews.ftr', use_threads = True)
users = pd.read_feather (FILE_PATH + 'users.ftr', use_threads = True)
products = pd.read_feather (FILE_PATH + 'products.ftr', use_threads = True)
products_brand_rank = pd.read_feather (FILE_PATH + 'products_brand_rank.ftr', use_threads = True)
# product_categories = pd.read_feather (FILE_PATH + 'product_categories.ftr', use_threads = True)

glowpick_before_labeling = pd.read_feather (FILE_PATH + 'glowpick_before_labeling.ftr', use_threads = True)
glowpick = pd.read_feather (FILE_PATH + 'glowpick.ftr', use_threads = True)

# products와 products_brand_rank를 merge한 dataframe
refined_products = pd.read_feather (FILE_PATH + 'refined_products.ftr', use_threads = True)

sparse_features = ["product_id", "user_id", "gender", "age", "skin_type", "idThirdCategory", ]
target = ['rating']

# 2.count #unique features for each sparse field and generate feature config for sequence feature

fixlen_feature_columns = [SparseFeat (feat, glowpick [feat].nunique(), embedding_dim = 4)
                          for feat in sparse_features]

linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names (linear_feature_columns + dnn_feature_columns)

# linear_feature_columns list 와 dnn_feature_columns list 를 load
with open(FILE_PATH + 'linear_feature_columns_list.pickle', 'rb') as fp:
    linear_feature_columns = pickle.load(fp)

with open(FILE_PATH + 'dnn_feature_columns_list.pickle', 'rb') as fp:
    dnn_feature_columns = pickle.load(fp)

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train, test = train_test_split (glowpick, test_size = 0.2)
train_model_input = {name: train [name] for name in feature_names}
test_model_input = {name: test [name] for name in feature_names}

device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# epoch 6
from math import sqrt
pred_ans = model.predict (test_model_input, batch_size = 256)

print("test MSE", round (mean_squared_error (test [target].values, pred_ans), 4))
print ("\ntest RMSE", round (sqrt (mean_squared_error (test [target].values, pred_ans)), 4))

use_col = ['created_at', 'rating', 'origin_user_id', 'origin_product_id', 'origin_age', 'origin_gender', 'price', 'brandName', 'origin_idThirdCategory', 'origin_skin_type']

# REAL_USER_ID = '1150641'
# top_n = 10
#
# real_model_input, new_glowpick = recommend.generate_user_input(REAL_USER_ID, glowpick, refined_products)
# top_n_recommend, top_n_real_reviews = recommend.recommendation(REAL_USER_ID, model, real_model_input, glowpick,
#                                                                new_glowpick, top_n)
#
#
# print('real_model_input',real_model_input)
from member.ctr.examples.shopngine import recommend

def Deep_FM_Recommend(REAL_USER_ID,top_n):

    real_model_input, new_glowpick = recommend.generate_user_input(REAL_USER_ID, glowpick, refined_products)
    top_n_recommend,top_n_real_reviews = recommend.recommendation(REAL_USER_ID,model,real_model_input,glowpick,new_glowpick,top_n)

    return top_n_recommend

print(Deep_FM_Recommend(119763,5))
print(Deep_FM_Recommend(24862,5))
