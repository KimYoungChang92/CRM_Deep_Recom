import pandas as pd
import scipy
import pandas as pd
from IPython.core.display import display, HTML
import re

from sklearn.manifold import TSNE
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

from sklearn import metrics

import matplotlib.pyplot as plt

display(HTML("<style>.container { width: 80% !important; }</style>"))
pd.set_option('display.max.colwidth',100)

ORDER_DATA_PATH = '/home/youngchang/1.Pyrepo/1.recom/1.작업폴더/Young_Jooyon_이걸로 수정하세요/joo_data/new/ing/jyshop_order_list.xlsx'
GOODS_DATA_PATH = '/home/youngchang/1.Pyrepo/1.recom/1.작업폴더/Young_Jooyon_이걸로 수정하세요/joo_data/new/ing/jyshop_goods_list_youngchang_copy_20200312.xlsx'
df_order = pd.read_excel(ORDER_DATA_PATH, sheet_name='Sheet1')

print(df_order.info())




