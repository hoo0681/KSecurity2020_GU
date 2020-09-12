class SO_img(FeatureType):
    """stream order 방식의 이미지 생성"""

    name = 'SO_img'
    dim = (256,256)

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
        SO.resize((256,256))
        return SO
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        