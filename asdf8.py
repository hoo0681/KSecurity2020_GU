import numpy as np
import re
class RawString(FeatureType):
    ''' Base class from which each feature type may inherit '''
    
    name = 'RawString'
    max_len=4000
    dim = (max_len,)
    types=np.unicode_
    PAD_First=True
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        ''' Generate a JSON-able representation of the file '''
        allstrings = map(bytes.decode,re.compile(b'[\x20-\x7f]{5,}').findall(bytez))#최소 5자이상의 문자열 추출
        return allstrings

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        if self.PAD_First:
            padded=[['<PAD>']*(self.max_len-len(raw_obj))+ raw_obj if len(raw_obj)<self.max_len else raw_obj[:self.max_len]][0]
        else:
            padded=[raw_obj+['<PAD>']*(self.max_len-len(raw_obj)) if len(raw_obj)<self.max_len else raw_obj[:self.max_len]][0]
        return np.array(padded,dtype=self.types)
        

    def feature_vector(self, bytez, lief_and_pefile):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(bytez, lief_and_pefile))
