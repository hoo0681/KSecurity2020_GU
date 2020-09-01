from sklearn.preprocessing import MinMaxScaler,minmax_scale\
    ,MaxAbsScaler,StandardScaler,RobustScaler,Normalizer,\
        QuantileTransformer,PowerTransformer
import pandas as pd
import numpy as np
class MalwareImg_IC(FeatureType):
    """ 설명"""

    name = 'MalwareImg_IC'
    dim = 256*256*10

    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        distributions = [
            ['Unscaled data', X],
            ['log scaled data',np.log(X+1)],
            ['Data after standard scaling', StandardScaler().fit_transform(X)],
            ['Data after min-max scaling', MinMaxScaler().fit_transform(X)],
            ['Data after max-abs scaling', MaxAbsScaler().fit_transform(X)],
            ['Data after robust scaling', RobustScaler(quantile_range=(25, 75)).fit_transform(X)],
            ['Data after power transformation (Yeo-Johnson)',PowerTransformer(method='yeo-johnson').fit_transform(X)],
            ['Data after quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal').fit_transform(X)],
            ['Data after quantile transformation (uniform pdf)',QuantileTransformer(output_distribution='uniform').fit_transform(X)],
            ['Data after sample-wise L2 normalizing',Normalizer().fit_transform(X)],
        ]
#inofzip=np.transpose(np.stack([ i[1] for i in distributions]))
        inofzip=(np.stack([ i[1] for i in distributions]))
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return inofzip

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        raise
        #return 모델에_넘겨줄_최종_데이터





        