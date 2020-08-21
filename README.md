# Reference
https://github.com/endgameinc/ember  

<br />

H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.  

```
@ARTICLE{2018arXiv180404637A,  
  author = {{Anderson}, H.~S. and {Roth}, P.},  
  title = "{EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models}",  
  journal = {ArXiv e-prints},  
  archivePrefix = "arXiv",  
  eprint = {1804.04637},  
  primaryClass = "cs.CR",  
  keywords = {Computer Science - Cryptography and Security},  
  year = 2018,  
  month = apr,  
  adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180404637A},  
}  
```  

<br />

# State Change
1. ember/features.py: change row variables -2018.10  
2. remove resource directory -2018.10  
3. change script files -2018.10  
4. add 01_extract.py, 02_train.py, 03_predict.py, 04_get_accuracy.py  -2018.10   
(this refer to ember/init.py, ember/features.py)
5. add utils directory  -2018.10 
6. add Test directory  -2018.10 
7. add output directory -2018.12   
8. add multiprocess job of extracting freature - 2019.01
9. Failed to develop multiprocess predcit. The AI framework developer ban it. - 2019.01
10. add pyqt gui(feature extraction)
    *change ember package. I reivse PEFeatureExtractor class in ember package(all methods in class have features argument. original code have not features argument)
11. remove console python module and add add gui module in core directory
12. change directory name 'ember' to 'core'

<br />
<br />

# Install
Above python 3.5    
```
;Install virtualenv
$ virtualenv env -p python3
$ . ./env/bin/activate
```
  
```
;Install python modules
(env)$ pip install -r requirements.txt
```

# Prerequisite
* inputfile(csv including label) structure without column's names  

![traindata_label](screenshot/traindata_label.png)