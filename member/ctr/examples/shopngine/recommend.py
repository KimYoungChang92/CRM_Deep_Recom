#!/home/lionkim/anaconda3/envs/torch2/bin/python
import pandas as pd
import numpy as np


def generate_user_input (REAL_USER_ID, glowpick, refined_products):
    REAL_USER_ID = str(REAL_USER_ID)
    feature_names = ['product_id', 'user_id', 'gender', 'age', 'skin_type', 'idThirdCategory']

    show_cols = ['created_at', 'rating', 'productTitle', 'volume', 'price', 'brandName',]

    sparse_features = ["product_id", "user_id", "gender", "age", "skin_type", "idThirdCategory", ]
    origin_feature_list = ['origin_product_id',
     'origin_user_id',
     'origin_gender',
     'origin_age',
     'origin_skin_type',
     'origin_idThirdCategory']

    used_features = sparse_features + origin_feature_list + show_cols

    real_method_df = glowpick [used_features]

    real_user_df = real_method_df [real_method_df ['origin_user_id'] == REAL_USER_ID]

    real_user_df2 = glowpick [glowpick ['origin_user_id'] == REAL_USER_ID]



    predicted_product_list = real_user_df ['origin_product_id'].values.tolist ()


    refined_products [refined_products ['product_id'] == '100020']


    review_product_id_list = glowpick ['origin_product_id'].unique()


    """ glowpick 은 결측치를 삭제했으므로 refined_products 보다 상품수가 적음
     len (glow_product_id_list) : 50761
     len (refined_products) : 86184
     glowpick 에 있는 상품만 가져옴 """
    removed_products = refined_products [refined_products ['product_id'].isin (review_product_id_list)]

    review_product_list = removed_products ['product_id'].values 

    non_review_product_list = np.setdiff1d (review_product_list, predicted_product_list)

    glowpick [glowpick ['origin_product_id'] == 0]

    new_glow = glowpick [glowpick ['origin_product_id'].isin (non_review_product_list)]

    new_glow.drop_duplicates (['origin_product_id'], inplace = True)

    new_features = ["product_id", "idThirdCategory", 'origin_product_id', 'productTitle', 'origin_idThirdCategory']
    new_glow [new_features]

    real_user_info_df = glowpick [glowpick ['origin_user_id'] == REAL_USER_ID].drop_duplicates ('origin_user_id')
    real_user_info_df = real_user_info_df [['user_id', 'gender', 'age', 'skin_type', 'origin_user_id']]

    new_glow1 = new_glow [new_features]

    new_glow1.reset_index (drop = True, inplace = True)
    real_user_info_df.reset_index (drop = True, inplace = True)

    new_glowpick = pd.concat ([new_glow1, real_user_info_df], axis = 1)

    user_data_dict = dict (zip (real_user_info_df.columns.tolist(), real_user_info_df.values.squeeze().tolist ()))
    new_glowpick.fillna (user_data_dict, inplace = True)
    new_glowpick

    real_model_input = {name: new_glowpick [name] for name in feature_names}
    
    return real_model_input, new_glowpick


def recommendation (REAL_USER_ID, model, real_model_input, glowpick, new_glowpick, top_n = 10, real_top_n = 20):

    pred_ans = model.predict (real_model_input, batch_size = 256)
    pred_list = list (map (lambda x : round (x, 2), pred_ans.flatten ().tolist ()))   # 예측한 점수 리스트
    user_pred_df = pd.DataFrame (pred_list, columns = ['rating'])
    user_product_pred_df = pd.concat ([new_glowpick, user_pred_df], axis = 1)
    user_product_pred_df.sort_values ('rating', ascending = False, inplace = True)
    recommend = user_product_pred_df [['productTitle', 'rating']]
    top_n_recommend  = recommend [: top_n]

    top_n_real_reviews = glowpick [glowpick ['origin_user_id'] == REAL_USER_ID] [['productTitle', 'rating']].sort_values ('rating', ascending = False,) [: real_top_n]
    
    return top_n_recommend, top_n_real_reviews
































