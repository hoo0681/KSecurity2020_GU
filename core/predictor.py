"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
import tqdm
import core
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import OrderedDict
from . import utility
import pickle
import logging

logger = logging.getLogger('Testing')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

# file handler
f_handler = logging.FileHandler('file.log')
f_handler.setLevel(logging.ERROR)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

logger.addHandler(handler)
logger.addHandler(f_handler)

class Predictor:
    def __init__(self, modelpath, testdir, features, output):
        # load model with pickle to predict
        with open(modelpath, 'rb') as f:
            self.model = lgb.Booster(model_file=modelpath)
        self.testdir = testdir
        self.output = output
        self.features = features

    def run(self):
        y_pred = []
        name = []
        err = 0
        end = len(next(os.walk(self.testdir))[2])

        for sample in tqdm.tqdm(utility.directory_generator(self.testdir),ascii=True, total=end):
            fullpath = os.path.join(self.testdir, sample)

            if os.path.isfile(fullpath):
                binary = open(fullpath, "rb").read()
                name.append(sample)

                try:
                    y_pred.append(core.predict_sample(self.model, binary, self.features))           
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as e:
                    logger.error('{}: {} error is occuered'.format(sample, e))            
                    y_pred.append(0)
                    err += 1

        y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)
        series = OrderedDict([('hash', name),('y_pred', y_pred)])
        r = pd.DataFrame.from_dict(series)
        r.to_csv(self.output, index=False, header=None)

        logger.info('{} error is occured'.format(err))