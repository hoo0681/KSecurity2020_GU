#!/usr/bin/python
"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
업데이트 08-21 export, import 함수를 주소로 다루는 경우는 해당 주소를 기록하도록 변경
업데이트 08-21-2 parsing warning 추가
"""

''' Extracts some basic features from PE files. Many of the features
implemented have been used in previously published works. For more information,
check out the following resources:
* Schultz, et al., 2001: http://128.59.14.66/sites/default/files/binaryeval-ieeesp01.pdf
* Kolter and Maloof, 2006: http://www.jmlr.org/papers/volume7/kolter06a/kolter06a.pdf
* Shafiq et al., 2009: https://www.researchgate.net/profile/Fauzan_Mirza/publication/242084613_A_Framework_for_Efficient_Mining_of_Structural_Information_to_Detect_Zero-Day_Malicious_Portable_Executables/links/0c96052e191668c3d5000000.pdf
* Raman, 2012: http://2012.infosecsouthwest.com/files/speaker_materials/ISSW2012_Selecting_Features_to_Classify_Malware.pdf
* Saxe and Berlin, 2015: https://arxiv.org/pdf/1508.03096.pdf

It may be useful to do feature selection to reduce this set of features to a meaningful set
for your modeling problem.
'''

import re
import pefile
import lief
import hashlib
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import time
import pandas as pd
from . import utility
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler,minmax_scale\
    ,MaxAbsScaler,StandardScaler,RobustScaler,Normalizer,\
        QuantileTransformer,PowerTransformer
LIEF_MAJOR, LIEF_MINOR, _ = lief.__version__.split('.')
LIEF_EXPORT_OBJECT = int(LIEF_MAJOR) > 0 or ( int(LIEF_MAJOR)==0 and int(LIEF_MINOR) >= 10 )

FEATURE_TPYE_LIST=[
              'ByteHistogram',
              'ByteEntropyHistogram',
              'SectionInfo',
              'ImportsInfo',
              'ExportsInfo',
              'GeneralFileInfo',
              'HeaderFileInfo',
              'StringExtractor',
              "ParsingWarning",
              "IsPacked",
              'RichHeader',
              'DataDirectories',
              "IMG_IC_origin",
              "IMG_IC_log",
              "SO_img",
              "RawString"
              #"IMG_IC_standard_scaling",
              #"IMG_IC_MinMax_scaling",
              #"IMG_IC_MaxAbs_scaling",
              #"IMG_IC_Robust_scaling",
              #"IMG_IC_normal_QuantileTransformer",
              #"IMG_IC_uniform_QuantileTransformer",
]

class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = (0,)
    types=np.dtype
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = cls.__name__
    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, bytez, lief_and_pefile):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplemented)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplemented)

    def feature_vector(self, bytez, lief_and_pefile):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(bytez, lief_and_pefile))

class ByteHistogram(FeatureType):
    ''' Byte histogram (count + non-normalized) over the entire binary file '''

    name = 'ByteHistogram'
    dim = (256,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        return counts.tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized
class ByteEntropyHistogram(FeatureType):
    ''' 2d byte/entropy histogram based loosely on (Saxe and Berlin, 2015).
    This roughly approximates the joint probability of byte value and local entropy.
    See Section 2.1.1 in https://arxiv.org/pdf/1508.03096.pdf for more info.
    '''

    name = 'ByteEntropyHistogram'
    dim = (256,)
    types=np.float32
    def __init__(self, step=1024, window=2048):
        super(FeatureType, self).__init__()
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block):
        # coarse histogram, 16 bytes per bin
        c = np.bincount(block >> 4, minlength=16)  # 16-bin histogram
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        H = np.sum(-p[wh] * np.log2(
            p[wh])) * 2  # * x2 b.c. we reduced information by half: 256 bins (8 bits) to 16 bins (4 bits)

        Hbin = int(H * 2)  # up to 16 bins (max entropy is 8 bits)
        if Hbin == 16:  # handle entropy = 8.0 bits
            Hbin = 15

        return Hbin, c

    def raw_features(self, bytez, lief_and_pefile):
        output = np.zeros((16, 16), dtype=np.int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            Hbin, c = self._entropy_bin_counts(a)
            output[Hbin, :] += c
        else:
            # strided trick from here: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]

            # from the blocks, compute histogram
            for block in blocks:
                Hbin, c = self._entropy_bin_counts(block)
                output[Hbin, :] += c

        return output.flatten().tolist()

    def process_raw_features(self, raw_obj):
        counts = np.array(raw_obj, dtype=np.float32)
        sum = counts.sum()
        normalized = counts / sum
        return normalized
class SectionInfo(FeatureType):
    ''' Information about section names, sizes and entropy.  Uses hashing trick
    to summarize all this section info into a feature vector.
    '''

    name = 'SectionInfo'
    dim =( 5 + 50 + 50 + 50 + 50 + 50,)
    types=np.float32

    def __init__(self):
        super(FeatureType, self).__init__()

    @staticmethod
    def _properties(s):
        return [str(c).split('.')[-1] for c in s.characteristics_lists]

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,_=lief_and_pefile
        if lief_binary is None:
            return {"entry": "", "sections": []}

        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break

        raw_obj = {"entry": entry_section}
        raw_obj["sections"] = [{
            'name': s.name,
            'size': s.size,
            'entropy': s.entropy,
            'vsize': s.virtual_size,
            'props': self._properties(s)
        } for s in lief_binary.sections]
        return raw_obj

    def process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),  # total number of sections
            # number of sections with nonzero size
            sum(1 for s in sections if s['size'] == 0),
            # number of sections with an empty name
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([raw_obj['entry']]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)
class ImportsInfo(FeatureType):
    ''' Information about imported libraries and functions from the
    import address table.  Note that the total number of imported
    functions is contained in GeneralFileInfo.
    '''

    name = 'ImportsInfo'
    dim = (1280,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,_=lief_and_pefile
        imports = {}
        if lief_binary is None:
            return imports

        for lib in lief_binary.imports:
            if lib.name not in imports:
                imports[lib.name] = []  # libraries can be duplicated in listing, extend instead of overwrite

            # Clipping assumes there are diminishing returns on the discriminatory power of imported functions
            #  beyond the first 10000 characters, and this will help limit the dataset size
            #함수를 주소로 다루는 경우는 해당 주소를 기록하도록 변경
            imports[lib.name].extend(["ordinal" + str(entry.ordinal) if entry.is_ordinal else entry.name[:10000] for entry in lib.entries])

        return imports

    def process_raw_features(self, raw_obj):
        # unique libraries
        libraries = list(set([l.lower() for l in raw_obj.keys()]))
        libraries_hashed = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]

        # A string like "kernel32.dll:CreateFileMappingA" for each imported function
        imports = [lib.lower() + ':' + e for lib, elist in raw_obj.items() for e in elist]
        imports_hashed = FeatureHasher(1024, input_type="string").transform([imports]).toarray()[0]

        # Two separate elements: libraries (alone) and fully-qualified names of imported functions
        return np.hstack([libraries_hashed, imports_hashed]).astype(np.float32)
class ExportsInfo(FeatureType):
    ''' Information about exported functions. Note that the total number of exported
    functions is contained in GeneralFileInfo.
    '''

    name = 'ExportsInfo'
    dim = (128,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,_=lief_and_pefile
        if lief_binary is None:
            return []

        # Clipping assumes there are diminishing returns on the discriminatory power of exports beyond
        #  the first 10000 characters, and this will help limit the dataset size
        # 함수를 주소로 다루는 경우는 해당 주소를 기록하도록 변경
        clipped_exports = [export.name[:10000] if LIEF_EXPORT_OBJECT else export[:10000] for export in lief_binary.exported_functions]

        return clipped_exports

    def process_raw_features(self, raw_obj):
        exports_hashed = FeatureHasher(128, input_type="string").transform([raw_obj]).toarray()[0]
        return exports_hashed.astype(np.float32)
class GeneralFileInfo(FeatureType):
    ''' General information about the file '''

    name = 'GeneralFileInfo'
    dim = (10,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,_=lief_and_pefile
        if lief_binary is None:
            return {
                'size': len(bytez),
                'vsize': 0,
                'has_debug': 0,
                'exports': 0,
                'imports': 0,
                'has_relocations': 0,
                'has_resources': 0,
                'has_signature': 0,
                'has_tls': 0,
                'symbols': 0
            }

        return {
            'size': len(bytez),
            'vsize': lief_binary.virtual_size,
            'has_debug': int(lief_binary.has_debug),
            'exports': len(lief_binary.exported_functions),
            'imports': len(lief_binary.imported_functions),
            'has_relocations': int(lief_binary.has_relocations),
            'has_resources': int(lief_binary.has_resources),
            'has_signature': int(lief_binary.has_signature),
            'has_tls': int(lief_binary.has_tls),
            'symbols': len(lief_binary.symbols),
        }

    def process_raw_features(self, raw_obj):
        return np.asarray(
            [
                raw_obj['size'], raw_obj['vsize'], raw_obj['has_debug'], raw_obj['exports'], raw_obj['imports'],
                raw_obj['has_relocations'], raw_obj['has_resources'], raw_obj['has_signature'], raw_obj['has_tls'],
                raw_obj['symbols']
            ],
            dtype=np.float32)
class HeaderFileInfo(FeatureType):
    ''' Machine, architecure, OS, linker and other information extracted from header '''

    name = 'HeaderFileInfo'
    dim = (62,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,_=lief_and_pefile
        raw_obj = {}
        raw_obj['coff'] = {'timestamp': 0, 'machine': "", 'characteristics': []}
        raw_obj['optional'] = {
            'subsystem': "",
            'dll_characteristics': [],
            'magic': "",
            'major_image_version': 0,
            'minor_image_version': 0,
            'major_linker_version': 0,
            'minor_linker_version': 0,
            'major_operating_system_version': 0,
            'minor_operating_system_version': 0,
            'major_subsystem_version': 0,
            'minor_subsystem_version': 0,
            'sizeof_code': 0,
            'sizeof_headers': 0,
            'sizeof_heap_commit': 0
        }
        if lief_binary is None:
            return raw_obj

        raw_obj['coff']['timestamp'] = lief_binary.header.time_date_stamps
        raw_obj['coff']['machine'] = str(lief_binary.header.machine).split('.')[-1]
        raw_obj['coff']['characteristics'] = [str(c).split('.')[-1] for c in lief_binary.header.characteristics_list]
        raw_obj['optional']['subsystem'] = str(lief_binary.optional_header.subsystem).split('.')[-1]
        raw_obj['optional']['dll_characteristics'] = [
            str(c).split('.')[-1] for c in lief_binary.optional_header.dll_characteristics_lists
        ]
        raw_obj['optional']['magic'] = str(lief_binary.optional_header.magic).split('.')[-1]
        raw_obj['optional']['major_image_version'] = lief_binary.optional_header.major_image_version
        raw_obj['optional']['minor_image_version'] = lief_binary.optional_header.minor_image_version
        raw_obj['optional']['major_linker_version'] = lief_binary.optional_header.major_linker_version
        raw_obj['optional']['minor_linker_version'] = lief_binary.optional_header.minor_linker_version
        raw_obj['optional'][
            'major_operating_system_version'] = lief_binary.optional_header.major_operating_system_version
        raw_obj['optional'][
            'minor_operating_system_version'] = lief_binary.optional_header.minor_operating_system_version
        raw_obj['optional']['major_subsystem_version'] = lief_binary.optional_header.major_subsystem_version
        raw_obj['optional']['minor_subsystem_version'] = lief_binary.optional_header.minor_subsystem_version
        raw_obj['optional']['sizeof_code'] = lief_binary.optional_header.sizeof_code
        raw_obj['optional']['sizeof_headers'] = lief_binary.optional_header.sizeof_headers
        raw_obj['optional']['sizeof_heap_commit'] = lief_binary.optional_header.sizeof_heap_commit
        return raw_obj

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['coff']['timestamp'],
            FeatureHasher(10, input_type="string").transform([[raw_obj['coff']['machine']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['coff']['characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['subsystem']]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([raw_obj['optional']['dll_characteristics']]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[raw_obj['optional']['magic']]]).toarray()[0],
            raw_obj['optional']['major_image_version'],
            raw_obj['optional']['minor_image_version'],
            raw_obj['optional']['major_linker_version'],
            raw_obj['optional']['minor_linker_version'],
            raw_obj['optional']['major_operating_system_version'],
            raw_obj['optional']['minor_operating_system_version'],
            raw_obj['optional']['major_subsystem_version'],
            raw_obj['optional']['minor_subsystem_version'],
            raw_obj['optional']['sizeof_code'],
            raw_obj['optional']['sizeof_headers'],
            raw_obj['optional']['sizeof_heap_commit'],
        ]).astype(np.float32)
class StringExtractor(FeatureType):
    ''' Extracts strings from raw byte stream '''

    name = 'StringExtractor'
    dim =( 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1,)
    types=np.float32
    def __init__(self):
        super(FeatureType, self).__init__()
        # all consecutive runs of 0x20 - 0x7f that are 5+ characters
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        # occurances of the string 'C:\'.  Not actually extracting the path
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        self._registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        self._mz = re.compile(b'MZ')

    def raw_features(self, bytez, lief_and_pefile):

        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0

        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printabledist': c.tolist(),  # store non-normalized histogram
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        hist_divisor = float(raw_obj['printables']) if raw_obj['printables'] > 0 else 1.0
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            np.asarray(raw_obj['printabledist']) / hist_divisor, raw_obj['entropy'], raw_obj['paths'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['MZ']
        ]).astype(np.float32)
class ParsingWarning(FeatureType):
    ''' ParsingWarning over the entire binary file '''

    name = 'ParsingWarning'
    dim = (2,)
    types=np.float32
    def __init__(self):#FeatureType상속
        super(FeatureType, self).__init__()

    def raw_features(self, bytez, lief_and_pefile):
        #일련의 바이트를 정해진 타입에 맞게 잘라 배열로 변환후 빈도 계산
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return {
                'has_warning':1,
                'warnings':1
            }
        if len(pe.get_warnings())==0:
            return {
                'has_warning':0,
                'warnings':0
            }
        else:
            return {
                'has_warning':1,
                'warnings':len(pe.get_warnings())
            }
        
    def process_raw_features(self, raw_obj):
        return np.asarray([
            raw_obj['has_warning'],raw_obj['warnings']],dtype=self.types
        )
class IsPacked(FeatureType):
    """ packed 되었는지 추측 Packed_PE_File_Detection_for_Malware_Forensics 참고"""

    name = 'IsPacked'
    dim = (1,)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return 0
        #if pe.get_section_by_rva(pe.OPTIONAL_HEADER.AddressOfEntryPoint) != None:
        #    enter_sect=pe.get_section_by_rva(pe.OPTIONAL_HEADER.AddressOfEntryPoint)
        #    if  enter_sect.IMAGE_SCN_MEM_EXECUTE and enter_sect.IMAGE_SCN_MEM_WRITE and enter_sect.get_entropy() >=6.85:
        #        return 1
        #elif pe.is_dll() :
        #    for sect in pe.sections:
        #        if sect.IMAGE_SCN_MEM_EXECUTE and sect.get_entropy()>=6.85:
        #            return 1
        #else:
        #    for sect in pe.sections:
        #        if hasattr(sect,'IMAGE_SCN_MEM_EXECUTE') and hasattr(sect,'IMAGE_SCN_MEM_WRITE') and sect.get_entropy()>=6.85:
        #            return 1
        #return 0
        if lief_binary is None:
            return 0
        # properties of entry point, or if invalid, the first executable section
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint)
            if (lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in entry_section.characteristics_lists) and (lief.PE.SECTION_CHARACTERISTICS.MEM_WRITE in entry_section.characteristics_lists) and (entry_section.entropy>=6.85):
                return 1
        except lief.not_found:
            # bad entry point, let's find the first executable section
            for s in lief_binary.sections:
                if (lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists) and (lief.PE.SECTION_CHARACTERISTICS.MEM_WRITE in s.characteristics_lists) and (s.entropy>=6.85):
                    return 1
        return 0
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return np.array([raw_obj]).astype(self.types)
class RichHeader(FeatureType):
    """RichHeader정보 어떻게 벡터화 할지 정해지지 않음"""
    name = 'RichHeader'
    dim = (30,)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기
    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return 0
        info={}
        info['id']=[i.id for i in lief_binary.rich_header.entries]
        info['build_id']=[i.build_id for i in lief_binary.rich_header.entries]
        info['count']=[i.count for i in lief_binary.rich_header.entries]
        
        return info
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        result = []
        if raw_obj==0:
            return result
        else:
            for key in raw_obj.keys():
                result.append(np.pad(raw_obj[key],(0,10), 'constant', constant_values=0)[:10])
            result=np.concatenate(result, axis=0)
            return result.reshape(self.dim).astype(self.types)
class SO_img(FeatureType):
    """stream order 방식의 이미지 생성"""

    name = 'SO_img'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return 0
        source2=bytearray(bytez)
        h=w=np.floor(len(source2)**0.5)
        h,w =int(h)+1,int(w)+1
        SO=np.zeros((h,w))
        h,w=int(h)-1,int(w)-1
        i=0
        for x in range(w):
            for y in range(h):
            #print(x,y)
                if i<len(source2):
                    SO[x,y]=source2[i]
                else:
                    SO[x,y]=0
                i+=1
        SO=np.resize(SO,(256,256))
        return SO.astype(self.types)
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
def GenerateTime(lief_binary):
    if lief_binary is None:
        return time.strftime('%Y-%m', time.gmtime(0))
    fileheader = lief_binary.header
    timestamp = time.gmtime(fileheader.time_date_stamps)
    return time.strftime('%Y-%m', timestamp)
class IMG_IC_origin(FeatureType):
    """ 설명"""

    name = 'IMG_IC_origin'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[:-1:1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        #return {'malimg':X.astype(np.float32).tolist()}
        return X.astype(np.float32)
    def process_raw_features(self, raw_obj):#추출한 값 가공
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_log(FeatureType):
    """ 설명"""

    name = 'IMG_IC_log'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[:-1:1],source[1::1]):
            image[x,y]+=1
        output=np.log(image+1)
        
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        #return {'malimg':output.astype(np.float32).tolist()}#이 내용이 기록됨
        return output.astype(np.float32)
    def process_raw_features(self, raw_obj):#추출한 값 가공  train,pridict전에 해당 과정을 거치고 들어감
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_standard_scaling(FeatureType):
    """ 설명"""

    name = 'IMG_IC_standard_scaling'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=StandardScaler().fit_transform(X)        
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_MinMax_scaling(FeatureType):
    """ 설명"""

    name = 'IMG_IC_MinMax_scaling'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=MinMaxScaler().fit_transform(X)      
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_MaxAbs_scaling(FeatureType):
    """ 설명"""

    name = 'IMG_IC_MaxAbs_scaling'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=MaxAbsScaler().fit_transform(X)        
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_Robust_scaling(FeatureType):
    """ 설명"""

    name = 'IMG_IC_Robust_scaling'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=RobustScaler(quantile_range=(25, 75)).fit_transform(X)    
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_normal_QuantileTransformer(FeatureType):
    """ 설명"""

    name = 'IMG_IC_normal_QuantileTransformer'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=QuantileTransformer(output_distribution='normal',n_quantiles=256).fit_transform(X)      
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)

    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class IMG_IC_uniform_QuantileTransformer(FeatureType):
    """ 설명"""

    name = 'IMG_IC_uniform_QuantileTransformer'
    dim = (256,256)
    types=np.float32
    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        source=bytearray(bytez)
        image=np.zeros((256,256))
        for x,y in zip(source[::1],source[1::1]):
            image[x,y]+=1
        X=image#.astype(np.uint8)
        output=QuantileTransformer(output_distribution='uniform',n_quantiles=256).fit_transform(X)      
        #bytez => 파일내용 타입: byte 길이 : 가변적
        #lief_and_pefile => life와 pefile로 parse된 결과물, 타입: 튜플, 내용 : (life_binary,pe)
        return output.astype(np.float32)
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        #return 모델에_넘겨줄_최종_데이터
class DataDirectories(FeatureType):
    ''' Extracts size and virtual address of the first 15 data directories '''

    name = 'DataDirectories'
    dim = (15 * 2,)
    types=np.float32    
    def __init__(self):
        super(FeatureType, self).__init__()
        self._name_order = [
            "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
            "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE",
            "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER"
        ]

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        output = []
        if lief_binary is None:
            return output

        for data_directory in lief_binary.data_directories:
            output.append({
                "name": str(data_directory.type).replace("DATA_DIRECTORY.", ""),
                "size": data_directory.size,
                "virtual_address": data_directory.rva
            })
        return output

    def process_raw_features(self, raw_obj):
        features = np.zeros((2 * len(self._name_order),), dtype=np.float32)
        for i in range(len(self._name_order)):
            if i < len(raw_obj):
                features[2 * i] = raw_obj[i]["size"]
                features[2 * i + 1] = raw_obj[i]["virtual_address"]
        return features
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
        return list(allstrings)

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

class PEFeatureExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''
    def __init__(self, featurelist, dim=0):
        self.features = featurelist
        #if dim == 0:
        #    self.dim = sum([fe.dim for fe in self.features])
    def unpack(self,arg):
        (func,bytez,pe_info)=arg
        return func(bytez,pe_info)
        
    def raw_features(self, bytez):
        try:
            lief_binary = lief.PE.parse(list(bytez))
            pe=pefile.PE(data=bytez)
            
        except ( lief.bad_format,lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
            lief_binary=None
            pe=None
            #features = {"appeared" :None}
        except Exception as e:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise
        lief_and_pefile=(lief_binary,pe)
        features = {"appeared" : GenerateTime(lief_binary)}
        features.update({fe.name: fe.raw_features(bytez, lief_and_pefile) for fe in self.features})
        return features
    def dict2npdict(self, bytez):
        try:
            pe=pefile.PE(data=bytez)
            lief_binary = lief.PE.parse(list(bytez))
        except lief.read_out_of_bound:
            try:
                lief_binary=lief.parse( bytez[:-len(pe.get_overlay())])
            except Exception as e:
                print('with OOB',e)
        except ( lief.bad_format,lief.bad_file, lief.pe_error, lief.parser_error, RuntimeError) as e:
            lief_binary=None
            pe=None
            #features = {"appeared" :None}
        except Exception as e:  # everything else (KeyboardInterrupt, SystemExit, ValueError):
            raise
        lief_and_pefile=(lief_binary,pe)
        #features = {"appeared" : GenerateTime(lief_binary)}
        #features={}
        features={fe.name: fe.feature_vector(bytez, lief_and_pefile) for fe in self.features}
        return features
    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, bytez):
        return self.process_raw_features(self.raw_features(bytez))