class RichHeader(FeatureType):
    """RichHeader정보 어떻게 벡터화 할지 정해지지 않음"""

    name = 'RichHeader'
    dim = 10

    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return 0
        info={}
        list_=[]
        for i in lief_binary.rich_header.entries:
            info['id']=i.id
            info['build_id']=i.build_id 
            info['count']=i.count
            list_.append(info)
        return list_
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        