import os
import json
import jsonlines
from . import features

def checkNone(_str_value):
    """
    Check if string is None
    """
    if not _str_value:
        return 0
    if _str_value == 'None':
        return 0

    return 1

def directory_generator(datadir):
    """
    Os.listdir to iterator
    """
    for sample in os.listdir(datadir):
        yield sample


def readonelineFromjson(jsonlpath):
    """
    Return features in JSONL.
    """
    with jsonlines.open(jsonlpath) as reader:
        for obj in reader:
            del obj['label']
            del obj['sha256']
            del obj['appeared']
            
            return list(obj.keys())

class FeatureType:
    def __init__(self):
        self.names = {
            'histogram': features.ByteHistogram(),
            'byteentropy': features.ByteEntropyHistogram(),
            'section': features.SectionInfo(),
            'imports': features.ImportsInfo(),
            'exports': features.ExportsInfo(),
            'general': features.GeneralFileInfo(),
            'header': features.HeaderFileInfo(),
            'strings': features.StringExtractor()
        }

    def parsing(self, lists):
        """
        return feature object for extracting
        """
        featurelist = []

        for feature in lists:
            if feature in self.names:
                featurelist.append(self.names.get(feature))

        return featurelist