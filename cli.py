from core import features, PEFeatureExtractor, train_model, create_vectorized_features, predict_sample
from core import extractfeature, utility, trainer, predictor, evaluationor
import os
import sys
import tqdm
import jsonlines
import pandas as pd
import multiprocessing
import lightgbm as lgb
import logging
import logging.config
import argparse
logger = logging.getLogger('CLI')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    print("hoo's: revision NUMBER:2021.02.24 22:07")
    print("가천대학교 201935364 홍승후 k시큐리티 특징추출기")
    parser = argparse.ArgumentParser()
    help_="[필수] 할 행동을 결정하세요 선택지:['TrainExtFeat','TestExtFeat'] "
    parser.add_argument("-A","--action",choices=["TrainExtFeat","TestExtFeat"],required=True, help=help_ )
    help_ = "[필수] 출력 파일의 경로를 지정합니다. 확장자까지 입력해야합니다. 예: ./myfeature.h5"
    parser.add_argument("-OP","--outputPath",required=False, help=help_)
    help_ = "[필수] 대상 파일들이 있는 폴더의 경로를 지정합니다. 오직 PE파일만 인식하며 이 외에는 정상적으로 작동하지 않습니다. 예: ./sample_data/"
    parser.add_argument("-TD", "--targetfolder",required=False, help=help_)
    help_ = "[필수] TrainExtFeat를 선택한경우 필수적으로 필요합니다. 타입은 csv이여야 합니다. 예: ./label.csv"
    parser.add_argument("-LaDir", "--labelfile",required=False, help=help_)
    help_ = "[옵션] 추출할 특징을 선택합니다. 기본값은 all, 미선언시 all입니다 "
    parser.add_argument("-selFeat", "--selectFeature",nargs='+',choices=[*features.FEATURE_TPYE_LIST,'all','None_img','only_img'],default='all',required=False, help=help_)
    
    #args = parser.parse_args('-A TrainExtFeat -OP ../result/feature/0919_2.h5 -TD ../result/content/dataset -LaDir ../result/content/asdf.csv'.split())
    args=parser.parse_args()
    if args.action == "TrainExtFeat":
        if ((args.targetfolder is None) or (args.outputPath is None) or (args.labelfile is None)):
            parser.error("plase check argument: {}{}{}".format(args.targetfolder,args.outputPath,args.labelfile))
        else:
            action=args.action
            output=args.outputPath
            dateset=args.targetfolder
            labelPath=args.labelfile
            print(args.selectFeature)
            if 'all' in args.selectFeature:
                features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST]
            elif ('None_img' in args.selectFeature) or ('only_img' in args.selectFeature):
                if ('None_img' in args.selectFeature) and ('only_img' in args.selectFeature):
                    parser.error("plase check argument")
                elif 'None_img' in args.selectFeature:
                    features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST if f.lower().find('img')==-1]
                elif 'only_img' in args.selectFeature:
                    features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST if f.lower().find('img')!=-1]
            else:
                features=[getattr(features,f)() for f in args.selectFeature]
            #features = []
            #for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            #    func=getattr(features,v)
            #    r.append(func())
            extractor = extractfeature.Extractor(dateset, labelPath, output, features)
    elif args.action == "TestExtFeat":
        if (args.targetfolder is None or args.outputPath is None or args.labelfile is not None):
            parser.error("plase check argument: {}{}{}".format(args.targetfolder,args.outputPath,args.labelfile))
        else:
            action=args.action
            output=args.outputPath
            dateset=args.targetfolder
            if 'all' in args.selectFeature:
                features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST]
            elif ('None_img' in args.selectFeature) or ('only_img' in args.selectFeature):
                if ('None_img' in args.selectFeature) and ('only_img' in args.selectFeature):
                    parser.error("plase check argument")
                elif 'None_img' in args.selectFeature:
                    features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST if f.lower().find('img')==-1]
                elif 'only_img' in args.selectFeature:
                    features=[getattr(features,f)() for f in features.FEATURE_TPYE_LIST if f.lower().find('img')!=-1]
            else:
                features=[getattr(features,f)() for f in args.selectFeature]
            extractor = extractfeature.testExtractor(dateset, output, features)
    extractor.run()
    logger.info('******* Extracting Done ******* \n')