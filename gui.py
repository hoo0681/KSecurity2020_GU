"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)

2020-08-20 업데이트: 피쳐추출방법이 추가되면 자동으로 ui도 업데이트됩니다. 
core/features.py의 FEATURE_TPYE_LIST에 사용할 클래스 이름을 적어두면 되고 ui를 별도로 수정하는 일은 하지않아도 됩니다
추출하는 특징이 15개를 넘어가면 간격조정이 필요할 수 있습니다.
업데이트 이전 코드는 주석처리만 해놓았습니다. 
예:>
features.py >>

    FEATURE_TPYE_LIST=[
                'ByteHistogram',
                'ByteEntropyHistogram',
                'SectionInfo',
                'ImportsInfo',
                'ExportsInfo',
                'GeneralFileInfo',
                'HeaderFileInfo',
                'StringExtractor'
    ]
2020-08-26 업데이트 matplotlib가 backend로 tkinter를 사용하는데 곳곳에
 있는 multiprocessing과 충돌이 일어난것으로 보고 backend를 pyqt5로 변경함 
 또한 진행바를 ascii로 표시하게 하였음

추가예정: tensorflow model 가능하도록 세팅
사용 가능한 모델 종류가 ui에 자동 반영되도록
모델 평가자료 비교기능

2020-08-27 업데이트 pyqt를 사용했더니 심각한 오류가 발생해서 되돌림
멀티프로세스, 멀티스레드 적용함 에러는 raise되도록 함

2020-09-12 업데이트 저장파일의 형식을 hdf5로 전면 수정 메모리 부족현상 완화유도
단, 기존의 기능과 호환여부확인 안함 절대 주의!!
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QLabel, QPushButton, QCheckBox
from core import features, PEFeatureExtractor, train_model, create_vectorized_features, predict_sample
from core import extractfeature, utility, trainer, predictor, evaluationor
import os
import sys
import tqdm
import jsonlines
import pandas as pd
import multiprocessing
import lightgbm as lgb
import logging
import logging.config

numberOfprocessor = 4 # it is for multi-processing

logger = logging.getLogger('GUI')
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-5s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Ui_MainWindow(QWidget):
    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(880, 533)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralWidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 851, 451))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.trainsetlabel = QtWidgets.QLabel(self.tab)
        self.trainsetlabel.setGeometry(QtCore.QRect(10, 10, 67, 17))
        self.trainsetlabel.setObjectName("trainsetlabel")
        self.trainsetPathLabel = QtWidgets.QLabel(self.tab)
        self.trainsetPathLabel.setGeometry(QtCore.QRect(80, 10, 491, 16))
        self.trainsetPathLabel.setObjectName("trainsetPathLabel")
        self.TrainSetbutton = QtWidgets.QPushButton(self.tab)
        self.TrainSetbutton.setGeometry(QtCore.QRect(10, 30, 61, 27))
        self.TrainSetbutton.setObjectName("TrainSetbutton")
        self.trainsetcsvLabel = QtWidgets.QLabel(self.tab)
        self.trainsetcsvLabel.setGeometry(QtCore.QRect(10, 70, 101, 17))
        self.trainsetcsvLabel.setObjectName("trainsetcsvLabel")
        self.trainsetcsvPathLabel = QtWidgets.QLabel(self.tab)
        self.trainsetcsvPathLabel.setGeometry(QtCore.QRect(110, 70, 591, 17))
        self.trainsetcsvPathLabel.setObjectName("trainsetcsvPathLabel")
        self.TrainSetCSVbutton = QtWidgets.QPushButton(self.tab)
        self.TrainSetCSVbutton.setGeometry(QtCore.QRect(10, 90, 61, 27))
        self.TrainSetCSVbutton.setObjectName("TrainSetCSVbutton")
        #####################################################
        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='Fe'+v+'ChkBox'
            self.__setattr__(new_attr_name,QtWidgets.QCheckBox(self.tab))
            new_attr=self.__getattribute__(new_attr_name)
            new_attr.setGeometry(QtCore.QRect(30+(170*((i)//7)), 130+(20*((i+7)%7)), 151, 22))
            new_attr.setChecked(True)
            new_attr.setObjectName(new_attr_name)

        #self.FeBytehistogramChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeBytehistogramChkBox.setGeometry(QtCore.QRect(30, 150, 151, 22))
        #self.FeBytehistogramChkBox.setChecked(True)
        #self.FeBytehistogramChkBox.setObjectName("FeBytehistogramChkBox")

        #self.FeByteEntropyHistogramChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeByteEntropyHistogramChkBox.setGeometry(QtCore.QRect(30, 180, 191, 22))
        #self.FeByteEntropyHistogramChkBox.setChecked(True)
        #self.FeByteEntropyHistogramChkBox.setObjectName("FeByteEntropyHistogramChkBox")

        #self.FeStringChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeStringChkBox.setGeometry(QtCore.QRect(30, 210, 151, 22))
        #self.FeStringChkBox.setChecked(True)
        #self.FeStringChkBox.setObjectName("FeStringChkBox")

        #self.FeGeneralChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeGeneralChkBox.setGeometry(QtCore.QRect(220, 150, 151, 22))
        #self.FeGeneralChkBox.setChecked(True)
        #self.FeGeneralChkBox.setObjectName("FeGeneralChkBox")

        #self.FeHeaderChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeHeaderChkBox.setGeometry(QtCore.QRect(220, 180, 141, 22))
        #self.FeHeaderChkBox.setChecked(True)
        #self.FeHeaderChkBox.setObjectName("FeHeaderChkBox")

        #self.FeSectionChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeSectionChkBox.setGeometry(QtCore.QRect(220, 210, 121, 22))
        #self.FeSectionChkBox.setChecked(True)
        #self.FeSectionChkBox.setObjectName("FeSectionChkBox")
        #######################################################
        self.ExtractOutputbutton = QtWidgets.QPushButton(self.tab)
        self.ExtractOutputbutton.setGeometry(QtCore.QRect(20, 320, 99, 27))
        self.ExtractOutputbutton.setObjectName("ExtractOutputbutton")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(20, 300, 67, 17))
        self.label_3.setObjectName("label_3")
        self.ExtractRunButton = QtWidgets.QPushButton(self.tab)
        self.ExtractRunButton.setGeometry(QtCore.QRect(670, 330, 151, 41))
        self.ExtractRunButton.setObjectName("ExtractRunButton")
        #######################################################
        #self.FeImportChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeImportChkBox.setGeometry(QtCore.QRect(380, 150, 131, 22))
        #self.FeImportChkBox.setChecked(True)
        #self.FeImportChkBox.setObjectName("FeImportChkBox")
        #self.FeExportsChkBox = QtWidgets.QCheckBox(self.tab)
        #self.FeExportsChkBox.setGeometry(QtCore.QRect(380, 180, 111, 22))
        #self.FeExportsChkBox.setChecked(True)
        #self.FeExportsChkBox.setObjectName("FeExportsChkBox")
        ########################################################
        self.line_2 = QtWidgets.QFrame(self.tab)
        self.line_2.setGeometry(QtCore.QRect(0, 120, 851, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.tab)
        self.line_3.setGeometry(QtCore.QRect(0, 270, 851, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.ExtractOutputPath = QtWidgets.QLabel(self.tab)
        self.ExtractOutputPath.setGeometry(QtCore.QRect(80, 300, 480, 17))
        self.ExtractOutputPath.setObjectName("ExtractOutputPath")
        # Train Panel
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.jsonlpath = QtWidgets.QLabel(self.tab_2)
        self.jsonlpath.setGeometry(QtCore.QRect(40, 30, 67, 17))
        self.jsonlpath.setObjectName("jsonlpath")
        self.jsonlPathLabel = QtWidgets.QLabel(self.tab_2)
        self.jsonlPathLabel.setGeometry(QtCore.QRect(120, 30, 480, 17))
        self.jsonlPathLabel.setObjectName("jsonlPathLabel")
        self.JsonOpenButton = QtWidgets.QPushButton(self.tab_2)
        self.JsonOpenButton.setGeometry(QtCore.QRect(40, 50, 99, 27))
        self.JsonOpenButton.setObjectName("JsonOpenButton")
        self.TrainButton = QtWidgets.QPushButton(self.tab_2)
        self.TrainButton.setGeometry(QtCore.QRect(660, 310, 151, 41))
        self.TrainButton.setObjectName("TrainButton")
        self.GradientBoostChk = QtWidgets.QCheckBox(self.tab_2)
        self.GradientBoostChk.setGeometry(QtCore.QRect(50, 110, 151, 22))
        self.GradientBoostChk.setChecked(True)
        self.GradientBoostChk.setObjectName("GradientBoostChk")
        self.line = QtWidgets.QFrame(self.tab_2)
        self.line.setGeometry(QtCore.QRect(0, 90, 851, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line2 = QtWidgets.QFrame(self.tab_2)
        self.line2.setGeometry(QtCore.QRect(0, 200, 851, 16))
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.trainoutput = QtWidgets.QLabel(self.tab_2)
        self.trainoutput.setGeometry(QtCore.QRect(40, 230, 67, 17))
        self.trainoutput.setObjectName("trainoutput")
        self.trainoutputLabel = QtWidgets.QLabel(self.tab_2)
        self.trainoutputLabel.setGeometry(QtCore.QRect(110, 230, 300, 17))
        self.trainoutputLabel.setObjectName("trainoutputLabel")
        self.trainoutputButton = QtWidgets.QPushButton(self.tab_2)
        self.trainoutputButton.setGeometry(QtCore.QRect(40, 250, 99, 27))
        self.trainoutputButton.setObjectName("trainoutputButton")
        # Test Panel
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.ModelPathButton = QtWidgets.QPushButton(self.tab_3)
        self.ModelPathButton.setGeometry(QtCore.QRect(30, 40, 99, 27))
        self.ModelPathButton.setObjectName("ModelPathButton")
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setGeometry(QtCore.QRect(30, 20, 67, 17))
        self.label.setObjectName("label")
        self.ModelPathLabel = QtWidgets.QLabel(self.tab_3)
        self.ModelPathLabel.setGeometry(QtCore.QRect(80, 20, 480, 17))
        self.ModelPathLabel.setObjectName("ModelPathLabel")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 91, 17))
        self.label_2.setObjectName("label_2")
        self.TestDatasetsPathLabel = QtWidgets.QLabel(self.tab_3)
        self.TestDatasetsPathLabel.setGeometry(QtCore.QRect(120, 80, 480, 17))
        self.TestDatasetsPathLabel.setObjectName("TestDatasetsPathLabel")
        self.TestDataSetPathButton = QtWidgets.QPushButton(self.tab_3)
        self.TestDataSetPathButton.setGeometry(QtCore.QRect(30, 100, 99, 27))
        self.TestDataSetPathButton.setObjectName("TestDataSetPathButton")
        self.TestFeatures = QtWidgets.QLabel(self.tab_3)
        self.TestFeatures.setGeometry(QtCore.QRect(30, 140, 71, 17))
        self.TestFeatures.setObjectName("TestFeatures")
        self.TestFeaturesLabel = QtWidgets.QLabel(self.tab_3)
        self.TestFeaturesLabel.setGeometry(QtCore.QRect(110, 140, 480, 17))
        self.TestFeaturesLabel.setObjectName("label_5")
        self.TestFeaturesButton = QtWidgets.QPushButton(self.tab_3)
        self.TestFeaturesButton.setGeometry(QtCore.QRect(30, 160, 99, 27))
        self.TestFeaturesButton.setObjectName("TestFeaturesButton")

        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(30, 200, 61, 17))
        self.label_4.setObjectName("label_4")
        self.TestOutputPath = QtWidgets.QLabel(self.tab_3)
        self.TestOutputPath.setGeometry(QtCore.QRect(90, 200, 480, 17))
        self.TestOutputPath.setObjectName("label_5")
        self.TestOutputButton = QtWidgets.QPushButton(self.tab_3)
        self.TestOutputButton.setGeometry(QtCore.QRect(30, 220, 99, 27))
        self.TestOutputButton.setObjectName("TestOutputButton")
        self.TestRunButton = QtWidgets.QPushButton(self.tab_3)
        self.TestRunButton.setGeometry(QtCore.QRect(668, 306, 131, 41))
        self.TestRunButton.setObjectName("TestRunButton")
        # Evaluation Panel
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.label_6 = QtWidgets.QLabel(self.tab_4)
        self.label_6.setGeometry(QtCore.QRect(40, 40, 91, 17))
        self.label_6.setObjectName("label_6")
        self.Testcsvpathlabel = QtWidgets.QLabel(self.tab_4)
        self.Testcsvpathlabel.setGeometry(QtCore.QRect(140, 40, 380, 16))
        self.Testcsvpathlabel.setObjectName("label_7")
        self.TestcsvPathButton = QtWidgets.QPushButton(self.tab_4)
        self.TestcsvPathButton.setGeometry(QtCore.QRect(40, 60, 99, 27))
        self.TestcsvPathButton.setObjectName("TestcsvPathButton")
        self.label_8 = QtWidgets.QLabel(self.tab_4)
        self.label_8.setGeometry(QtCore.QRect(40, 110, 67, 17))
        self.label_8.setObjectName("label_8")
        self.csvlabelpath = QtWidgets.QLabel(self.tab_4)
        self.csvlabelpath.setGeometry(QtCore.QRect(110, 110, 380, 17))
        self.csvlabelpath.setObjectName("label_9")
        self.LabelPathButton = QtWidgets.QPushButton(self.tab_4)
        self.LabelPathButton.setGeometry(QtCore.QRect(40, 130, 99, 27))
        self.LabelPathButton.setObjectName("LabelPathButton")
        self.EvaluationRunButton = QtWidgets.QPushButton(self.tab_4)
        self.EvaluationRunButton.setGeometry(QtCore.QRect(678, 346, 141, 41))
        self.EvaluationRunButton.setObjectName("EvaluationRunButton")
        self.tabWidget.addTab(self.tab_4, "")
        
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.testsetlabel = QtWidgets.QLabel(self.tab_5)
        self.testsetlabel.setGeometry(QtCore.QRect(10, 10, 67, 17))
        self.testsetlabel.setObjectName("testsetlabel")
        self.testsetPathLabel = QtWidgets.QLabel(self.tab_5)
        self.testsetPathLabel.setGeometry(QtCore.QRect(80, 10, 491, 16))
        self.testsetPathLabel.setObjectName("testsetPathLabel")
        self.TestSetbutton = QtWidgets.QPushButton(self.tab_5)
        self.TestSetbutton.setGeometry(QtCore.QRect(10, 30, 61, 27))
        self.TestSetbutton.setObjectName("TestSetbutton")

        #####################################################
        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='testFe'+v+'ChkBox'
            self.__setattr__(new_attr_name,QtWidgets.QCheckBox(self.tab_5))
            new_attr=self.__getattribute__(new_attr_name)
            new_attr.setGeometry(QtCore.QRect(30+(170*((i)//7)), 130+(20*((i+7)%7)), 151, 22))
            new_attr.setChecked(True)
            new_attr.setObjectName(new_attr_name)

        self.testExtractOutputbutton = QtWidgets.QPushButton(self.tab_5)
        self.testExtractOutputbutton.setGeometry(QtCore.QRect(20, 320, 99, 27))
        self.testExtractOutputbutton.setObjectName("testExtractOutputbutton")
        self.testlabel_3 = QtWidgets.QLabel(self.tab_5)
        self.testlabel_3.setGeometry(QtCore.QRect(20, 300, 67, 17))
        self.testlabel_3.setObjectName("testlabel_3")
        self.testExtractRunButton = QtWidgets.QPushButton(self.tab_5)
        self.testExtractRunButton.setGeometry(QtCore.QRect(670, 330, 151, 41))
        self.testExtractRunButton.setObjectName("testExtractRunButton")

        self.testline_2 = QtWidgets.QFrame(self.tab_5)
        self.testline_2.setGeometry(QtCore.QRect(0, 120, 851, 16))
        self.testline_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.testline_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.testline_2.setObjectName("testline_2")
        self.testline_3 = QtWidgets.QFrame(self.tab_5)
        self.testline_3.setGeometry(QtCore.QRect(0, 270, 851, 16))
        self.testline_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.testline_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.testline_3.setObjectName("testline_3")
        self.testExtractOutputPath = QtWidgets.QLabel(self.tab_5)
        self.testExtractOutputPath.setGeometry(QtCore.QRect(80, 300, 480, 17))
        self.testExtractOutputPath.setObjectName("testExtractOutputPath")

        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 880, 25))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
        transform widgets of Pyqt5
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # Extract features panel
        self.trainsetlabel.setText(_translate("MainWindow", "TrainSet:"))
        self.trainsetPathLabel.setText(_translate("MainWindow", "None"))
        self.TrainSetbutton.setText(_translate("MainWindow", "Open"))
        self.trainsetcsvLabel.setText(_translate("MainWindow", "TrainSetLabel:"))
        self.trainsetcsvPathLabel.setText(_translate("MainWindow", "None"))
        self.TrainSetCSVbutton.setText(_translate("MainWindow", "Open"))
        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='Fe'+v+'ChkBox'
            new_attr=self.__getattribute__(new_attr_name)
            new_attr.setText(_translate("MainWindow", v))
        #self.FeBytehistogramChkBox.setText(_translate("MainWindow", "ByteHistogram"))#151
        #self.FeByteEntropyHistogramChkBox.setText(_translate("MainWindow", "ByteEntropyHistogram"))#191
        #self.FeStringChkBox.setText(_translate("MainWindow", "StringExtractor"))#151
        #self.FeGeneralChkBox.setText(_translate("MainWindow", "GeneralFileInfo"))#151
        #self.FeHeaderChkBox.setText(_translate("MainWindow", "HeaderFileInfo"))#141
        #self.FeSectionChkBox.setText(_translate("MainWindow", "SectionInfo"))#121
        self.ExtractOutputbutton.setText(_translate("MainWindow", "Open"))
        self.label_3.setText(_translate("MainWindow", "output:"))
        self.ExtractRunButton.setText(_translate("MainWindow", "Run"))
        #self.FeImportChkBox.setText(_translate("MainWindow", "ImportsInfo"))#141
        #self.FeExportsChkBox.setText(_translate("MainWindow", "ExportsInfo"))#111
        self.ExtractOutputPath.setText(_translate("MainWindow", "None"))
        # Train panel
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Extract features"))
        self.jsonlpath.setText(_translate("MainWindow", "jsonlpath:"))
        self.jsonlPathLabel.setText(_translate("MainWindow", "None"))
        self.JsonOpenButton.setText(_translate("MainWindow", "Open"))
        self.TrainButton.setText(_translate("MainWindow", "Run"))
        self.GradientBoostChk.setText(_translate("MainWindow", "GradientBoost"))
        self.trainoutput.setText(_translate("MainWindow", "output:"))
        self.trainoutputLabel.setText(_translate("MainWindow", "None"))
        self.trainoutputButton.setText(_translate("MainWindow", "Open"))
        # Test Panel
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Train"))
        self.ModelPathButton.setText(_translate("MainWindow", "Open"))
        self.label.setText(_translate("MainWindow", "Model:"))
        self.ModelPathLabel.setText(_translate("MainWindow", "None"))
        self.label_2.setText(_translate("MainWindow", "TestDataset:"))
        self.TestDatasetsPathLabel.setText(_translate("MainWindow", "None"))
        self.TestDataSetPathButton.setText(_translate("MainWindow", "Open"))
        self.label_4.setText(_translate("MainWindow", "output:"))
        self.TestFeatures.setText(_translate("MainWindow", "jsonlpath:"))
        self.TestFeaturesLabel.setText(_translate("MainWindow", "None"))
        self.TestFeaturesButton.setText(_translate("MainWindow", "Open"))
        self.TestOutputPath.setText(_translate("MainWindow", "None"))
        self.TestOutputButton.setText(_translate("MainWindow", "Open"))
        self.TestRunButton.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Test"))
        # Evaluation Panel
        self.label_6.setText(_translate("MainWindow", "Testcsvpath:"))
        self.Testcsvpathlabel.setText(_translate("MainWindow", "None"))
        self.TestcsvPathButton.setText(_translate("MainWindow", "Open"))
        self.label_8.setText(_translate("MainWindow", "labelPath:"))
        self.csvlabelpath.setText(_translate("MainWindow", "None"))
        self.LabelPathButton.setText(_translate("MainWindow", "Open"))
        self.EvaluationRunButton.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Evaluation"))
        
        self.testsetlabel.setText(_translate("MainWindow", "TestSet:"))
        self.testsetPathLabel.setText(_translate("MainWindow", "None"))
        self.TestSetbutton.setText(_translate("MainWindow", "Open"))

        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='testFe'+v+'ChkBox'
            new_attr=self.__getattribute__(new_attr_name)
            new_attr.setText(_translate("MainWindow", v))
        self.testExtractOutputbutton.setText(_translate("MainWindow", "Open"))
        self.testlabel_3.setText(_translate("MainWindow", "output:"))
        self.testExtractRunButton.setText(_translate("MainWindow", "Run"))

        self.testExtractOutputPath.setText(_translate("MainWindow", "None"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "test Extract features"))
    def check_extractfeatures(self):
        """
        features.py의 메소드를 로드하고 자동으로 추가해주도록 변경하자
        Check Features list in extract panel.
        this features used extracting
        """
        r = []
        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='Fe'+v+'ChkBox'
            new_attr=self.__getattribute__(new_attr_name)
            if new_attr.isChecked():
                func=getattr(features,v)
                r.append(func())


        #if self.FeBytehistogramChkBox.isChecked(): #ByteHistogram
        #    r.append(features.ByteHistogram())
        #if self.FeByteEntropyHistogramChkBox.isChecked(): #ByteEntropyHistogram
        #    r.append(features.ByteEntropyHistogram())
        #if self.FeStringChkBox.isChecked(): # String
        #    r.append(features.StringExtractor())
        #if self.FeGeneralChkBox.isChecked(): # GeneralFileInfo
        #    r.append(features.GeneralFileInfo())
        #if self.FeHeaderChkBox.isChecked(): # HeaderFileInfo
        #    r.append(features.HeaderFileInfo())
        #if self.FeSectionChkBox.isChecked(): # SectionInfo
        #    r.append(features.SectionInfo())
        #if self.FeImportChkBox.isChecked(): # ImportInfo
        #    r.append(features.ImportsInfo())
        #if self.FeExportsChkBox.isChecked(): # ExportInfo
        #    r.append(features.ExportsInfo())

        return r
    def check_testextractfeatures(self):
        """
        features.py의 메소드를 로드하고 자동으로 추가해주도록 변경하자
        Check Features list in extract panel.
        this features used extracting
        """
        r = []
        for i,v in zip(range(len(features.FEATURE_TPYE_LIST)),features.FEATURE_TPYE_LIST):
            new_attr_name='testFe'+v+'ChkBox'
            new_attr=self.__getattribute__(new_attr_name)
            if new_attr.isChecked():
                func=getattr(features,v)
                r.append(func())
        return r

    def clearvalue(self):
        self.trainsetPathLabel.setText("")
    def testclearvalue(self):
        self.testsetPathLabel.setText("")

    """
    widget events
    """
    def connect_event(self):
        """
        Initalize the events
        """
        #Extract features
        self.TrainSetbutton.clicked.connect(self.TrainSetbutton_click)
        self.TrainSetCSVbutton.clicked.connect(self.TrainSetcsvbutton_click)
        self.ExtractRunButton.clicked.connect(self.extractBtn_click)
        self.ExtractOutputbutton.clicked.connect(self.ExtractOutputbutton_click)
        
        #Train
        self.JsonOpenButton.clicked.connect(self.JsonOpenButton_click)
        self.trainoutputButton.clicked.connect(self.trainoutputButton_click)
        self.TrainButton.clicked.connect(self.TrainButton_click)

        #Test
        self.ModelPathButton.clicked.connect(self.ModelPathButton_click)
        self.TestDataSetPathButton.clicked.connect(self.TestDataSetPathButton_click)
        self.TestOutputButton.clicked.connect(self.TestOutputButton_click)
        self.TestFeaturesButton.clicked.connect(self.TestFeaturesButton_click)
        self.TestRunButton.clicked.connect(self.TestRunButton_click)

        #Evaluation
        self.TestcsvPathButton.clicked.connect(self.TestcsvPathButton_click)
        self.LabelPathButton.clicked.connect(self.LabelPathButton_click)
        self.EvaluationRunButton.clicked.connect(self.EvaluationRunButton_click)
        #test feature
        self.TestSetbutton.clicked.connect(self.TestSetbutton_click)
        self.testExtractRunButton.clicked.connect(self.testextractBtn_click)
        self.testExtractOutputbutton.clicked.connect(self.testExtractOutputbutton_click)
    def TrainSetbutton_click(self):
        """
        Choose TrainSetPath(directory)
        """
        DirName = QFileDialog.getExistingDirectory(self, "Select Folder")
        # To do : add Exception
        if DirName:
            self.trainsetPathLabel.setText(DirName)
        
    def TestSetbutton_click(self):
        """
        Choose TestSetPath(directory)
        """
        DirName = QFileDialog.getExistingDirectory(self, "Select Folder")
        # To do : add Exception
        if DirName:
            self.testsetPathLabel.setText(DirName)
        
    def TrainSetcsvbutton_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv)", options=options)
        if fileName:
            self.trainsetcsvPathLabel.setText(fileName)

    def ExtractOutputbutton_click(self):
        """
        Set output label in a extract panel 
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "","h5 Files (*.h5)", options=options)
        if fileName:
            self.ExtractOutputPath.setText(fileName)
    def testExtractOutputbutton_click(self):
        """
        Set output label in a extract panel 
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "","h5 Files (*.h5)", options=options)
        if fileName:
            self.testExtractOutputPath.setText(fileName)
    def ModelPathButton_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","txt Files (*.txt)", options=options)
        if fileName:
            self.ModelPathLabel.setText(fileName)

    def JsonOpenButton_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Jsonl Files (*.jsonl)", options=options)
        if fileName:
            self.jsonlPathLabel.setText(fileName)

    def trainoutputButton_click(self):
        """
        Choose TrainSetPath(directory)
        """
        DirName = QFileDialog.getExistingDirectory(self, "Select Folder")
        # To do : add Exception
        if DirName:
            self.trainoutputLabel.setText(DirName)

    def TestDataSetPathButton_click(self):
        """
        Choose TestSsetPath
        """
        DirName = QFileDialog.getExistingDirectory(self, "Select Folder")
        # To do : add Exception
        if DirName:
            self.TestDatasetsPathLabel.setText(DirName)

    def TestOutputButton_click(self):
        """
        Set output label in a extract panel 
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv)", options=options)
        if fileName:
            self.TestOutputPath.setText(fileName)
    
    def TestFeaturesButton_click(self):
        """
        Set TestSet features(jsonl)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","jsonl Files (*.jsonl)", options=options)
        if fileName:
            self.TestFeaturesLabel.setText(fileName)

    """
    Evaluation Panel Event
    """
    def TestcsvPathButton_click(self):
        """
        Set csvOftest label in a evaluation panel 
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv)", options=options)
        if fileName:
            self.Testcsvpathlabel.setText(fileName)

    def LabelPathButton_click(self):
        """
        Set labelOftest label in a evaluation panel 
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv)", options=options)
        if fileName:
            self.csvlabelpath.setText(fileName)

    """
    Run Button events
    """
    def extractBtn_click(self):
        """
        Run Button in a 'Extract' panel.
        """
        # Extract features which user select       
        features = self.check_extractfeatures()
        trainsetpath = self.trainsetPathLabel.text()
        trainsetlabelpath = self.trainsetcsvPathLabel.text()
        output = self.ExtractOutputPath.text()

        if not utility.checkNone(trainsetpath):
            logger.info('[Error] Please check the trainsetpath')
            return
        if not utility.checkNone(trainsetlabelpath):
            logger.info('[Error] Please check the labelpath')
            return
        if not utility.checkNone(output):
            output = os.path.join(os.getcwd(), 'features.jsonl')           

        extractor = extractfeature.Extractor(trainsetpath, trainsetlabelpath, output, features)
        extractor.run()

        logger.info('******* Extracting Done ******* \n')
    def testextractBtn_click(self):
        """
        Run Button in a 'Extract' panel.
        """
        # Extract features which user select       
        features = self.check_extractfeatures()
        testsetpath = self.testsetPathLabel.text()
        output = self.testExtractOutputPath.text()

        if not utility.checkNone(testsetpath):
            logger.info('[Error] Please check the trainsetpath')
            return
        if not utility.checkNone(output):
            output = os.path.join(os.getcwd(), 'features.h5')           

        extractor = extractfeature.testExtractor(testsetpath, output, features)
        extractor.run()

        logger.info('******* Extracting Done ******* \n')


    def TrainButton_click(self):
        """
        Run Button on a 'train' Tab.
        """
        output = self.trainoutputLabel.text()
        jsonlpath = self.jsonlPathLabel.text()

        if not utility.checkNone(jsonlpath):
            logger.info('[Error] Please check the jsonpath')
            return
        if not  utility.checkNone(output):
            logger.debug("Bingo {0} {1}".format(output, os.getcwd()))
            output = os.getcwd()

        logger.debug("jsonlpath {0}, output {1}".format(jsonlpath, output))

        train = trainer.Trainer(jsonlpath, output)
        train.run()
        
        logger.info('******* Training Done ******* \n')

    def TestRunButton_click(self, parameter_list):
        modelpath = self.ModelPathLabel.text()
        testdir = self.TestDatasetsPathLabel.text()
        output = self.TestOutputPath.text()
        featurelist = self.TestFeaturesLabel.text()
        features = utility.readonelineFromjson(featurelist)
        feature_parser = utility.FeatureType()
        featureobjs = feature_parser.parsing(features)

        if not utility.checkNone(modelpath):
            logger.info('[Info] Please check the modelpath')
            return
        if not utility.checkNone(testdir):
            logger.info('[Info] Please check the testdirpath')
            return
        if not utility.checkNone(output):
            output = os.path.join(os.getcwd(), 'result.csv') 
        
        logger.info('[Info] The list of features in selected JSONL file')
        logger.info('Total fatures : {}'.format(len(features)))
        [logger.info(featurename) for featurename in features]

        predict = predictor.Predictor(modelpath, testdir, featureobjs, output)
        predict.run()

        logger.info('******* Testing Done ******* \n')

    def EvaluationRunButton_click(self):
        testcsvpath = self.Testcsvpathlabel.text()
        testlabelpath = self.csvlabelpath.text()

        if not utility.checkNone(testcsvpath):
            logger.info('[Error] Please check the testcsvpath')
            return
        if not utility.checkNone(testlabelpath):
            logger.info('[Error] Please check the testlabelpath')
            return
        
        evaluate = evaluationor.Evaluator(testcsvpath, testlabelpath)
        evaluate.run()

if __name__ == "__main__":
    print("hoo's: revision NUMBER:2020.10.20 15:37")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.connect_event()

    MainWindow.show()
    sys.exit(app.exec_())