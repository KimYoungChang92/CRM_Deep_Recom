#!/home/lionkim/anaconda3/envs/torch2/bin/python
import pandas as pd
import pickle
import ast

FILE_PATH = '/home/lionkim/data_set/1.HM/glowpick/'

# 상품 DataFrame 에서 title이 중복된 row중 리뷰수가 50보다 작은 row를 삭제하기 위해 
# 상품 DataFrame 에 중복된 title이 있음
# 해당 아이디 리스트를 리턴
def delete_dupli_under_review_cnt (df, count = 50):
    df_dupl = df.groupby ('productTitle').filter (lambda x : len (x) > 2)

    # rank 와 reviewCnt 형변환. object to int64
    type_change_cols = ['rank']

    df_dupl [type_change_cols] = df_dupl [type_change_cols].astype ('int')

    # 리뷰수가 50보다 작은 row는 삭제할 것임
    df_dupl = df_dupl [df_dupl ['reviewCount'] < count]

    return df_dupl ['product_id'].values.tolist ()


# 상품 DataFrame에서 중복된 title은 title 뒤에 언더바('_')와 product_id를 붙여서 다시저장
def transform_ (df_products, duplication_title_list):
    df_products ['productTitle'] = df_products.apply (lambda row : row ['productTitle'] + '_' + str (row ['product_id']) 
                             if row ['productTitle'] in duplication_title_list 
                             else row ['productTitle'], axis = 1)
    return df_products


"""
    products : 상품 dataframe. (brand와 rank)가 없음
    products_brand_rank:  brand와 rank가 있는 상품 dataframe
"""
def refine_products (products, products_brand_rank):
    # products_brand_rank 중간 중간에 columns 들이 row 값으로 들어가있어서 
    # 해당 row를 빼고 새로운 데이터프레임 생성
    products_brand_rank = products_brand_rank [products_brand_rank ['rank'] != 'rank']
    # products_df.info ()
    
    # Merge product and products_brand_rank
    merged_products = pd.merge (products, products_brand_rank, how = 'left', on = 'product_id')
    
    # 일단 keywords, colorType은 삭제
    merged_products.drop (['keywords', 'colorType'], axis = 1, inplace = True)

    # 'price', 'thirdCategoryText', 'rank', 'brand'에서 결측치가 있는 row 삭제
    merged_products.dropna (subset = ['price', 'thirdCategoryText', 'rank', 'brand',], how = 'any', axis = 0, inplace = True)
    
    # 딕셔너리처럼 생긴 스트링 타입 brand 컬럼에서 brandTtile을 가져와서 brandName 이라는 컬럼을 생성
    merged_products ['brandName'] = merged_products ['brand'].apply (lambda brand : ast.literal_eval (brand) ['brandTitle'])

    # rank 와 reviewCnt 형변환. object to int64
    int_cols = ['rank']

    merged_products [int_cols] = merged_products [int_cols].astype ('int')
    # merged_products.dtypes

    # # merged_products dataframe에서 title이 중복된 것중 review수 (reviewCnt)가 50이하인 상품을 삭제하기 위해
    # # 해당 product_id list를 생성
    # del_id_list = delete_dupli_under_review_cnt (merged_products)

    # len (del_id_list)

    # with open (FILE_PATH + 'del_id_list', 'wb') as fp:
    #     pickle.dump (del_id_list, fp)


    # merged_products dataframe에서 title이 중복된 것중 review수 (reviewCnt)가 50이하인 상품을 삭제하기 위해
    # 해당 product_id list를 생성

    # 삭제할 product_id list를 불러옴
    with open (FILE_PATH + 'del_id_list', 'rb') as fp:
        del_id_list = pickle.load (fp)

    print ("len del_id_list: ", len (del_id_list), "\n")

    # productTitle이 중복되어 삭제할 product_id list에 포함되지 않은 row들을 가져옴
    products_refined = merged_products [~merged_products ['product_id'].isin (del_id_list)]


    # title이 동일한 row가 아직 존재함.
    # products_refined ['productTitle'].value_counts ()

    # print (len (products_df) - len (del_id_list))
    # print (len (products_refined))

    # products_refined ['productTitle'].value_counts 으로된 dataframe을 생성
    df_prod_ref_counts = products_refined ['productTitle'].value_counts().to_frame()

    # df_prod_ref_counts [df_prod_ref_counts ['productTitle'] > 1]

    # tilte이 둘 이상인 title 리스트 
    duplication_title_list = df_prod_ref_counts [df_prod_ref_counts ['productTitle'] >= 2].index.tolist()

    # 상품 DataFrame에서 중복된 title은 title 뒤에 언더바('_')와 product_id를 붙여서 다시저장한 DataFrame을 반환
    return transform_ (products_refined, duplication_title_list)
































