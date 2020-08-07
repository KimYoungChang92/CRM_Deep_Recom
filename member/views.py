# from keras.applications import inception_v3
# from keras.preprocessing import image


from django.shortcuts import render
import argparse
import json
import numpy as np
import requests

from django.http import HttpResponse
from numpy.core.defchararray import strip


import json
from django.core import serializers
from django.http import JsonResponse


##### reco_model
# from member.reco_model.wide_deep import *
# from member.reco_model.Model_utils.data_utils import *
# from member.reco_model.Model_utils.torch_model import *
# from member.reco_model.Model_utils.customer_multi import *

###### Python Import


from member.ctr.examples.shopngine.product_refine import *
from member.ctr.examples.shopngine.recommend import *
from member.ctr.examples.shopngine.common import *
from member.ctr.examples.shopngine.req_recom import *

def torch_predict(request):
    with open('/home/youngchang/PycharmProjects/CRM_dir2/rest_server/member/reco_model/test.json', 'r') as json_file:
        json_data = json.load(json_file)

    #content_type=u"application/json; charset=utf-8"

    return JsonResponse(json_data,json_dumps_params={'ensure_ascii' :False})

# ====================== wide and deep ====================
def predict_rating(request, user_id, top_n):

    get_customer = multi_customer(user_id,top_n)
    json_load_v1 = get_customer.to_json(orient='columns', force_ascii=False) # orient='columns' , orient='records'
    json_load_v2 = json.loads(json_load_v1)


    return JsonResponse(json_load_v2, json_dumps_params={'ensure_ascii': False} , safe=False)

# ====================== DeepFM ====================

def deepfm(request,REAL_USER_ID,top_n):
    get_customer = Deep_FM_Recommend(REAL_USER_ID,top_n)

    json_load_v1 = get_customer.to_json(orient='columns', force_ascii=False) # orient='columns' , orient='records'
    json_load_v2 = json.loads(json_load_v1)


    return JsonResponse(json_load_v2, json_dumps_params={'ensure_ascii': False} , safe=False)


