"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
from core import PEFeatureExtractor
import tqdm
import jsonlines
import pandas as pd
import multiprocessing
from . import utility
import logging
import gc
import lief
import pefile
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

#console handler
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

# add logging handler
logger.addHandler(handler)
logger.addHandler(f_handler)

class Extractor:
    def __init__(self, datadir, label, output, features):
        self.datadir = datadir
        self.output = output
        self.data = pd.read_csv(label, names=['hash', 'y'])
        self.features = features
    def extract_features(self, sample):
        """
        Extract features.
        If error is occured, return None Object
        """
        extractor = PEFeatureExtractor(self.features)
        fullpath = os.path.join(os.path.join(self.datadir, sample))
        try:
            binary = open(fullpath, 'rb').read()
            #feature = extractor.raw_features(binary)
            feature = extractor.dict2npdict(binary)
            feature.update({"sha256": sample}) # sample name(hash)
            feature.update({"label" :self.data[self.data.hash==sample].values[0][1]}) #label

        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:  
            logger.error('{}: {} error is occuered'.format(sample, e))
            #raise
            return None

        return feature

    def extract_unpack(self, args):
        """
        Pass thorugh function unpacking arguments
        """
        idx,path=args
        return (idx,self.extract_features(path))

    def extractor_multiprocess(self):
        """
        Ready to do multi Process
        Note that total variable in tqdm.tqdm should be revised
        Currently, I think that It is not safely. Because, multiprocess pool try to do FILE I/O.
        """
        end = len(next(os.walk(self.datadir))[2])
        extractor_iterator = ((idx,sample) for idx, sample in enumerate(utility.directory_generator(self.datadir)))
        try:
            #print(self.output)
            datasetF=h5py.File(self.output, 'r+')
            print(datasetF.keys())
            filename_set=datasetF['sha256']
            label_set=datasetF['label']
            feature_set_dict={fe.name:datasetF[fe.name] for fe in self.features}
        except (Exception , OSError) as e:
            print('new file',self.output)
            datasetF= h5py.File(self.output, 'w')
            dt=h5py.string_dtype()
            filename_set=datasetF.create_dataset('sha256',(0,),dtype=dt,maxshape=(None,),chunks=True)
            label_set=datasetF.create_dataset('label',(0,),dtype=np.uint8,maxshape=(None,),chunks=True)
            feature_set_dict={fe.name:datasetF.create_dataset(fe.name,(0,*fe.dim),dtype=fe.types,maxshape=(None,*fe.dim),chunks=True) for fe in self.features}
            #else:
            #    raise e
        firstidx=filename_set.shape[0]

        filename_set.resize((filename_set.shape[0]+end,))
        label_set.resize((label_set.shape[0]+end,))
        for i in feature_set_dict.values():
            i.resize((i.shape[0]+end,*i.shape[1:]))
        print(end)
        print(filename_set.shape)
        print(label_set.shape)
        try:
            with ProcessPoolExecutor(max_workers=4) as pool:
                with tqdm.tqdm(total=end,ascii=True,position=0, leave=True,desc='feature progress') as progress:
                    with tqdm.tqdm(total=end,ascii=True,position=1, leave=True,desc='save progress') as save_progress:
                        futures = []
                        for file in extractor_iterator:
                            future = pool.submit(self.extract_unpack, file)
                            future.add_done_callback(lambda p: self.progress_print(progress,p))
                            futures.append(future)
                        #print('done')
                        for f in as_completed(futures):
                            idx,result = f.result()
                            for k,i in result.items():
                                if k =='sha256':
                                    filename_set[firstidx+idx,...]=i
                                    save_progress.set_postfix_str("{}".format(i))
                                elif k =='label':
                                    label_set[firstidx+idx,...]=int(i)
                                else:
                                    feature_set_dict[k][firstidx+idx,...]=i
                            if f.done():
                                futures.remove(f)
                                save_progress.update(1)
        except Exception as e:
            print('error: ',e)
            filename_set.resize((firstidx,))
            label_set.resize((firstidx,))
            for i in feature_set_dict.values():
                i.resize((firstidx,*i.shape[1:]))
            datasetF.close()
            pass
        except:
            datasetF.close()
            raise
        datasetF.close()
        print('GC start')
        gc.collect()
        print('GC done')
    def progress_print(self, progress,future):
        _,i=future.result()
        progress.set_postfix_str("{}".format(i['sha256']))
        progress.update(1)
    def run(self):
        self.extractor_multiprocess()
    

class testExtractor:
    def __init__(self, datadir, output, features):
        self.datadir = datadir
        self.output = output
        self.features = features
    def extract_features(self, sample):
        """
        Extract features.
        If error is occured, return None Object
        """
        extractor = PEFeatureExtractor(self.features)
        fullpath = os.path.join(os.path.join(self.datadir, sample))
        try:
            binary = open(fullpath, 'rb').read()
            #feature = extractor.raw_features(binary)
            feature = extractor.dict2npdict(binary)
            feature.update({"sha256": sample}) # sample name(hash)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:  
            logger.error('{}: {} error is occuered'.format(sample, e))
            #raise
            return None

        return feature

    def extract_unpack(self, args):
        """
        Pass thorugh function unpacking arguments
        """
        idx,path=args
        return (idx,self.extract_features(path))

    def extractor_multiprocess(self):
        """
        Ready to do multi Process
        Note that total variable in tqdm.tqdm should be revised
        Currently, I think that It is not safely. Because, multiprocess pool try to do FILE I/O.
        """
        end = len(next(os.walk(self.datadir))[2])
        extractor_iterator = ((idx,sample) for idx, sample in enumerate(utility.directory_generator(self.datadir)))
        try:
            datasetF=h5py.File(self.output, 'r+')
            filename_set=datasetF['sha256']
            feature_set_dict={fe.name:datasetF[fe.name] for fe in self.features}
        except (Exception , OSError) as e:
            print('new file',self.output)
            datasetF= h5py.File(self.output, 'w')
            dt=h5py.string_dtype()
            filename_set=datasetF.create_dataset('sha256',(0,),dtype=dt,maxshape=(None,),chunks=True)
            feature_set_dict={fe.name:datasetF.create_dataset(fe.name,(0,*fe.dim),dtype=fe.types,maxshape=(None,*fe.dim),chunks=True) for fe in self.features}
            #else:
            #    raise e
        firstidx=filename_set.shape[0]

        filename_set.resize((filename_set.shape[0]+end,))
        for i in feature_set_dict.values():
            i.resize((i.shape[0]+end,*i.shape[1:]))
        try:
            with ProcessPoolExecutor(max_workers=4) as pool:
                with tqdm.tqdm(total=end,ascii=True,position=0, leave=True,desc='feature progress') as progress:
                    with tqdm.tqdm(total=end,ascii=True,position=1, leave=True,desc='save progress') as save_progress:
                        futures = []
                        for file in extractor_iterator:
                            future = pool.submit(self.extract_unpack, file)
                            future.add_done_callback(lambda p: self.progress_print(progress,p))
                            futures.append(future)
                        #print('done')
                        for f in as_completed(futures):
                            idx,result = f.result()
                            for k,i in result.items():
                                if k =='sha256':
                                    filename_set[firstidx+idx,...]=i
                                    save_progress.set_postfix_str("{}".format(i))
                                else:
                                    feature_set_dict[k][firstidx+idx,...]=i
                            if f.done():
                                futures.remove(f)
                                save_progress.update(1)
        except Exception as e:
            print('error: ',e)
            filename_set.resize((firstidx,))
            for i in feature_set_dict.values():
                i.resize((firstidx,*i.shape[1:]))
            datasetF.close()
            pass
        except:
            datasetF.close()
            raise
        datasetF.close()
        print('GC start')
        gc.collect()
        print('GC done')
    def progress_print(self, progress,future):
        _,i=future.result()
        progress.set_postfix_str("{}".format(i['sha256']))
        progress.update(1)
    def run(self):
        self.extractor_multiprocess()