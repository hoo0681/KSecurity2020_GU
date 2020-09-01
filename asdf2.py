from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback, sys

from sklearn.preprocessing import MinMaxScaler,minmax_scale\
    ,MaxAbsScaler,StandardScaler,RobustScaler,Normalizer,\
        QuantileTransformer,PowerTransformer
import pandas as pd
class MalwareImg_IC(FeatureType):
    """ 설명"""

    name = 'MalwareImg_IC'
    dim = 출력물의 차원수

    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return 가공되지 않은 추출된 값

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return 모델에_넘겨줄_최종_데이터





        