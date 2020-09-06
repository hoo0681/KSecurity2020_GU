import distorm3
class IsPacked(FeatureType):
    """ packed 되었는지 추측 Packed_PE_File_Detection_for_Malware_Forensics 참고"""

    name = 'IsPacked'
    dim = 1

    def __init__(self): #생성자
        super(FeatureType, self).__init__()#상속받기

    def raw_features(self, bytez, lief_and_pefile):
        lief_binary,pe=lief_and_pefile
        if (pe is None) or (lief_binary is None):
            return 0
        if pe.FILE_HEADER.Machine == 0x14c :
            mach_bit=distorm3.Decode32Bits
        elif pe.FILE_HEADER.Machine==0x200:
            mach_bit=distorm3.Decode64Bits
        else:
            return 0
        try:
            entry_section = lief_binary.section_from_offset(lief_binary.entrypoint).name
        except lief.not_found:
            # bad entry point, let's find the first executable section
            entry_section = ""
            for s in lief_binary.sections:
                if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in s.characteristics_lists:
                    entry_section = s.name
                    break
    def process_raw_features(self, raw_obj):#추출한 값 가공
        #raw_obj =>raw_features에서 반환하는 값
        #가공과정
        return raw_obj
        