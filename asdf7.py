import numpy as np
import os
import pandas as pd
import lief
import pefile
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import re

def getapis(arg):
    (filepath,val)=arg
    #tmp={}
    tmp=[]
    try:
        lief_bin=lief.parse(filepath)
    except lief.read_out_of_bound:
        try:
            lief_bin=lief.parse( bytearray(open(filepath,'rb').read())[:-len(pefile.PE(filepath,fast_load=True).get_overlay())])
        except:
            return (3, {'error':filepath})
    try:
        for lib in lief_bin.imports:
            for entry in lib.entries:
                if entry.is_ordinal:
                    api_name=lib.name+": ordinal " + str(entry.ordinal)
                else:
                    api_name=entry.name
                #tmp.setdefault(api_name,1)
                tmp.append(api_name)
    except:
        return (3, {'error':filepath})
    return (val,tmp)#Counter(tmp))

def getStrings(arg):
    (filepath,val)=arg
    allstrings = map(bytes.decode,re.compile(b'[\x20-\x7f]{5,}').findall(open(filepath,'rb').read()))#최소 5자이상의 문자열 추출
    return (val,allstrings)#Counter(allstrings))

def waiting(dicts,futures,name):
    errorFiles=[]
    for i in futures:
        val_,resu=i.result()
        if val_ == 3 :
            errorFiles.append(resu['error'])
        #dicts[val_]+=resu
        else:
            dicts[val_].extend(resu)
    return (dicts,errorFiles,name)
data_dir='/content/kisa_dataset/train_600/'
JH_api=[list(),list()]#[Counter(),Counter()]
JH_st=[list(),list()]#[Counter(),Counter()]
errorFiles=[]
api_futures=[]
string_futures=[]
writer_futures=[]
filters=re.compile(b'[\x20-\x7f]{5,}')

with ProcessPoolExecutor(max_workers=4) as pool:
    for filename,val in pd.read_csv('/content/kisa_dataset/train_600.csv',names=['file','val']).dropna().astype({'val':'int'}).values:
        future = pool.submit(getapis, (data_dir+filename,val))
        api_futures.append(future)
        future_ = pool.submit(getStrings, (data_dir+filename,val))
        string_futures.append(future_)
    dicts,errors,names=waiting(JH_api,api_futures,'apis')
    errorFiles.append(errors)
    if names=='apis':
        JH_api=dicts
    elif names=='string':
        JH_st=dicts
    dicts,errors,names=waiting(JH_st,string_futures,'string')
    errorFiles.append(errors)
    if names=='apis':
        JH_api=dicts
    elif names=='string':
        JH_st=dicts

JH_st_=[Counter(JH_st[0]),Counter(JH_st[1])]
JH_api_=[Counter(JH_api[0]),Counter(JH_api[1])]

print('악성API top 2000',JH_api_[1].most_common(2000))
print('악성API top 1000',JH_api_[1].most_common(1000))
print('정상API top 2000',JH_api_[0].most_common(2000))
print('정상API top 1000',JH_api_[0].most_common(1000))
print('악성string top 2000',JH_st_[1].most_common(2000))
print('악성string top 1000',JH_st_[1].most_common(1000))
print('정상string top 2000',JH_st_[0].most_common(2000))
print('정상string top 1000',JH_st_[0].most_common(1000))