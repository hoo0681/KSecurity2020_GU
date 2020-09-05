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
from concurrent.futures import ProcessPoolExecutor

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
            binary = open(fullpath, 'rb').read().close()
            #feature = extractor.raw_features(binary)
            feature = extractor.raw_features(binary)
            feature.update({"sha256": sample}) # sample name(hash)
            feature.update({"label" : self.data[self.data.hash==sample].values[0][1]}) #label

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
        return self.extract_features(args)

    def extractor_multiprocess(self):
        """
        Ready to do multi Process
        Note that total variable in tqdm.tqdm should be revised
        Currently, I think that It is not safely. Because, multiprocess pool try to do FILE I/O.
        """
        #pool = multiprocessing.Pool(4)
        #queue = multiprocessing.Queue()
        #queue.put('safe')
        end = len(next(os.walk(self.datadir))[2])
        #error = 0
        #tmp=[]
        pefiles=[]
        nonpefiles=[]
        for path, dir_, files in os.walk(self.datadir):
            for file in files:
                try:
                    lief.PE.parse(path+file)
                    pefile.PE(path+file)
                    #lief.PE.parse(file)
                    pefiles.append(file)
                except ( lief.bad_format,lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
                    nonpefiles.append(file)
                    #raise e
#
        #with jsonlines.open('./nonPefiles.json', 'w') as f:
        #    f.write(nonpefiles)
        print('non-pe file : {}, pe file : {}, total :{}'.format(len(nonpefiles),len(pefiles),end))
        extractor_iterator = ((sample) for idx, sample in enumerate(pefiles))
        #extractor_iterator = ((sample) for idx, sample in enumerate(utility.directory_generator(self.datadir)))
        
        with jsonlines.open(self.output, 'w') as f:
            with ProcessPoolExecutor(max_workers=4) as pool:
                with tqdm.tqdm(total=len(pefiles),ascii=True) as progress:
                    for x in pool.map(self.extract_unpack,extractor_iterator,chunksize=10):
                        f.write(x)
                #futures = []
                #pool.map(self.extract_unpack,extractor_iterator,chunksize=1000)
                #for file in extractor_iterator:
                #    future = pool.submit(self.extract_unpack, file)
                #    future.add_done_callback(lambda p: progress.update())
                #    futures.append(future)
#                results = []
                #for future_ in futures:
                #    result = future_.result()
                #    with jsonlines.open(self.output, 'a') as f:
                #        f.write(result)
#                        results.append(result)
        #    for x in tqdm.tqdm(pool.imap_unordered(self.extract_unpack, extractor_iterator),ascii=True, total=end):
        #        if not x:
        #            """
        #            To input error class or function
        #            """
        #            #raise
        #            error += 1
        #            continue
        #        #tmp.append(x)
        #        msg = queue.get()
        #        if msg == 'safe': 
        #            f.write(x)                
        #            queue.put('safe')
        #    #for item in tmp:
        #    #    f.write(item)
        #pool.close()
        #gclist=gc.get_stats()
        #for i in gclist:
        #    for key,val in i.items():
        #        print(key,': ',val)
        #    print("\n")
        print('GC start')
        gc.collect()
        print('GC done')
        #gclist=gc.get_stats()
        #for i in gclist:
        #    for key,val in i.items():
        #        print(key,': ',val)
        #    print("\n")
    def run(self):
        self.extractor_multiprocess()
        