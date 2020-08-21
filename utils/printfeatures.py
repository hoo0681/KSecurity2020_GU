import sys
import argparse
import os

#syspath = os.getcwd()
#sys.path.append(os.path.join(syspath, '../core'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from core import utility

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', help="jsonl path", required=True)
args = parser.parse_args()

featureobj = utility.FeatureType()
featurenames = utility.readonelineFromjson(args.src)

for idx, featurename in enumerate(featurenames):
    print('{}: {}'.format(idx, featurename))