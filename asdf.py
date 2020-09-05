import lief
import pandas as pd
PWD='./kisa_dataset/train_600/'
df= pd.read_csv('./kisa_dataset/train_600.csv')#csv는 pandas로 다룬다
df.columns=['file','val']
df.dropna(inplace=True)#없는 값 없애기
df.astype({'val':'int'})
target_list= df.values.tolist()
api_dict=[dict(),dict()]#val가 0이면 좌측 dict에 1이면 우측에 
error_msgs=['yara matching error','file peformat is something wrong ','pefile dumping error: {}','lief error']
pbar=tqdm.tqdm(total=len(df), position=0, leave=True,ascii=True)
for file_,val in target_list:#경로 폴더 파일 
  try:
    error_msg_idx=1#='file peformat is something wrong '
    lief_bin=lief.parse(PWD+file_)
    idx=int(val)#값을 인덱스로 활용
    #########DO SOMETHING###########
    for lib in lief_bin.imports:#import dll탐색
      for entry in lib.entries:#import 함수 탐색
        if entry.is_ordinal:#이름없이 주소로 호출되면
          api_name=lib.name+": ordinal " + str(entry.ordinal)
        else:#아니면
          api_name=entry.name
        if api_name not in api_dict[idx].keys():#처음보는 함수명이면
          api_dict[idx][api_name] = 1  #함수명을 key로 해서 새로 생성
        else:#아니면
          api_dict[idx][api_name] +=1 #1을 더한다
    #########DO SOMETHING###########
  except Exception as e:
    print('*may be something wrong in ',path+'/'+filename,e,' plase check')
    error_file_list.append((path+'/'+filename,error_msgs[error_msg_idx]))
  pbar.set_postfix_str('processed: %s' % (file_))
  pbar.update(1)
pbar.close()
test_df=pd.DataFrame.from_records(api_dict,index=['safe','Malware'])
test_df=test_df.fillna(0).T
#test_df.to_excel('asdf.xlsx')
test_df.to_csv('asdf.csv')