from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import joblib
import sklearn
import qimage2ndarray as qn
import test_segment as ts
import train_createmodel as tcm

import sys
import os, shutil

from MainWindow import Ui_MainWindow


class MyMessageBox(QMessageBox):
	def __init__(self):
		QMessageBox.__init__(self)
		self.setSizeGripEnabled (True)   

		self.setWindowTitle ('Model Paramethers')
		self.setText("Available models' paramethers for segmentation. \nCheck the table anytime, anywhere, by pressing Ctrl+t . \n \nIf your model isn't shown, check that it's saved on the rigth directory. \nIt should be under ~/RetinaPlus/Models")
		self.setIcon(QMessageBox.Information)
		self.setStandardButtons(QMessageBox.Ok)

		#Add TableWidget to QMessageBox           
		self.addTableWidget (self) 

		#Return values while clicking QPushButton        
		currentClick = self.exec_() 


	#Create TableWidget 
	def addTableWidget (self, parentItem):
		self.tableWidget = QTableWidget(parentItem)

		mypath_models=os.getcwd()+'/Models/'
		onlyfiles = [os.path.splitext(f)[0] for f in os.listdir(mypath_models) if os.path.isfile(os.path.join(mypath_models, f))]
		nr_row=len(onlyfiles)

		self.tableWidget.setRowCount(nr_row)
		self.tableWidget.setColumnCount(8)
		
		labelsH=['kernel', 'C', 'gamma', 'n_estimators', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'max_features']
		labelsV=onlyfiles
		self.tableWidget.setHorizontalHeaderLabels(labelsH)
		self.tableWidget.setVerticalHeaderLabels(labelsV)

		for i in range(0, nr_row):
			filename= os.getcwd()+'/Models/'+onlyfiles[i]+'.pkl'
			model = joblib.load(filename)
			params=model.get_params()
			if 'kernel' in params.keys():
				self.tableWidget.setItem(i,0, QTableWidgetItem(str(params.get('kernel'))))
				self.tableWidget.setItem(i,1, QTableWidgetItem(str(params.get('C'))))
				if params.get('kernel') == 'rbf':
					self.tableWidget.setItem(i,2, QTableWidgetItem(str(params.get('gamma'))))
				else:
					self.tableWidget.setItem(i,2, QTableWidgetItem('-'))
			else:
				for j in range(0, 3):
					self.tableWidget.setItem(i,j, QTableWidgetItem('-'))							

			if 'n_estimators' in params.keys():
				self.tableWidget.setItem(i,3, QTableWidgetItem(str(params.get('n_estimators'))))
				self.tableWidget.setItem(i,4, QTableWidgetItem(str(params.get('max_depth'))))
				self.tableWidget.setItem(i,5, QTableWidgetItem(str(params.get('min_samples_split'))))	
				self.tableWidget.setItem(i,6, QTableWidgetItem(str(params.get('max_leaf_nodes'))))
				self.tableWidget.setItem(i,7, QTableWidgetItem(str(params.get('max_features'))))		
			else:
				for j in range(3, 8):
					self.tableWidget.setItem(i,j, QTableWidgetItem('-'))



		self.tableWidget.move(40, 125)
		self.tableWidget.resize(520, 150)

	#Allow resizing of QMessageBox
	def event(self, e):
		result = QMessageBox.event(self, e)
		self.setMinimumWidth(0)
		self.setMaximumWidth(16777215)
		self.setMinimumHeight(0)
		self.setMaximumHeight(16777215)        
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.resize(600, 325)

		return result 

class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)

class Worker(QRunnable):
	def __init__(self, fn, *args):
		super(Worker, self).__init__()
		self.fn = fn
		self.args = args

		self.signals = WorkerSignals()

	@pyqtSlot()
	def run(self):
		print("Thread start")
		result=self.fn(*self.args)
		self.signals.result.emit(result)
		self.signals.finished.emit()


class MainWindow(QMainWindow, Ui_MainWindow):

	def __init__(self, *args, **kwargs):
		super(MainWindow, self).__init__(*args, **kwargs)

		self.setupUi(self)
		
		self.scaleFactor = 0.0
		
		self.createActions()
		self.createMenus()

		#Remove tabs: build model/use model, this way the tab only appears when selected in the comboBox
		self.tabWidget.removeTab(2)
		self.tabWidget.removeTab(1)

		#Get nr of saved models
		self.initialSavedModels = self.getModelList()
		self.initialNrSavedModels=len(self.initialSavedModels)


		#Choose a tab    
		self.comboBox.activated.connect(self.activated)

		#User View: action select image from file to segment
		self.pushButton_chooseImg.clicked.connect(self.chooseImg)
		self.pushButton_chooseImg.setShortcut(QKeySequence("Ctrl+i"))
		
		#User View: action enable segmentation button
		self.comboBox_chooseModel.activated.connect(self.choose_model_to_use)

		#User View: action start segmentation
		self.pushButton_segmentation.clicked.connect(self.thread_segmentar)
		
		#User View: action enable save button
		self.lineEdit_saveImg_seg.textChanged.connect(self.pre_save)
		self.lineEdit_saveImg_seg.textEdited.connect(self.pre_save)
		
		#User View: action save segmented image
		self.pushButton_saveImg.clicked.connect(self.save)
		
		#User Build: action select paramethers for the choosen classifier
		self.comboBox_selectMethod.activated.connect(self.choose_model_to_use_BUILD)
		
		#User Build: action import dataset
		self.pushButton_chooseDataset.clicked.connect(lambda: self.import_dataset('Images')) #lambda function: to pass an argument
		self.pushButton_chooseGT.clicked.connect(lambda: self.import_dataset('Ground Truth'))
		
		#User Build: enable train model button
		self.lineEdit_newModelName.textChanged.connect(self.writeNewModelName)
		self.lineEdit_newModelName.textEdited.connect(self.writeNewModelName)

		#User Build: train model
		self.pushButton_train.clicked.connect(self.thread_train)

		#User Build: select an image to show segmentation result
		self.pushButton_ImportImgTrain.clicked.connect(self.chooseTrainImg2testModel)
		self.pushButton_ImportImgTest.clicked.connect(self.chooseTestImg2testModel)
		
		#User Build: save created model and add to comboBoxChooseModel
		self.pushButton_saveModel.clicked.connect(self.saveNewModel)

		#Accept drag and drop
		self.setAcceptDrops(True)

		self.threadpool = QThreadPool()
		print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")

				
#Activate choosen view
	def activated(self, i):     # i is an int
		print('combo box choose tab activated:', i)
		
		if i==0:
			self.tabWidget.insertTab(-1, self.tab_useModel, "Use Model")
			self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab_useModel))

			self.updateComboBoxChooseModel() 			#add itens to comboBox.chooseModel


		elif i==1:
			self.tabWidget.insertTab(-1, self.tab_buildModel, "Build Model")
			self.tabWidget.setCurrentIndex(self.tabWidget.indexOf(self.tab_buildModel))

#User View: Update comboBoxChooseModel
	def getModelList(self):
		mypath_models=os.getcwd()+'/Models/'
		onlyfiles = [os.path.splitext(f)[0] for f in os.listdir(mypath_models) if os.path.isfile(os.path.join(mypath_models, f))]
		return onlyfiles

	def updateComboBoxChooseModel(self):
		onlyfiles = self.getModelList()
		self.comboBox_chooseModel.clear()
		self.comboBox_chooseModel.addItems(onlyfiles)
		#By default the app comes with a nr of models, if none other has been added has been added, combo box set to -1
		if len(onlyfiles)==self.initialNrSavedModels:
			self.comboBox_chooseModel.setCurrentIndex(-1)


#User View: Drag and Drop			
	def dragEnterEvent(self, event):
		print('drag')
		if event.mimeData().hasImage:
		   	 event.accept()
		   	 
		else:
			event.ignore()

	def dragMoveEvent(self, event):
		print('dragmove')
		if event.mimeData().hasImage:
			event.accept()
		else:
			event.ignore()

	def dropEvent(self, event):
		if self.tabWidget.currentIndex() == self.tabWidget.indexOf(self.tab_useModel):  #so faz drag and drop na label do use_model
			if event.mimeData().hasImage:
				print('drop')
				file_path = event.mimeData().urls()[0].toLocalFile()
				print('Drop sucessful in the chosen window!')
				self.pic(file_path)
				event.accept()
			else:
				event.ignore()
		else:
			event.ignore()


#User View: Choose image from file
	def chooseImg(self, s):
		print("click", s)
		dialog = QFileDialog(self)
		dialog.setWindowTitle("Choose an image pls!")
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setDirectory(os.path.expanduser("~")+'/Documents')
		dialog.setNameFilter("Images (*.png *.gif *.jpg *.tif)")
		dialog.setViewMode(QFileDialog.Detail)
		dialog.fileSelected.connect(self.pic)

		dialog.show()
		dialog.exec()


#User View: Show Chosen Image to segment 		
	def pic(self, x):
		print("Selected image:", x)
		self.imgToSegment = x
		image = QImage(x)

		self.label_showImgOrig = QLabel()
		self.label_showImgOrig.setBackgroundRole(QPalette.Base)
		self.label_showImgOrig.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.label_showImgOrig.setScaledContents(True)
		self.scrollArea_ImgOrig.setWidget(self.label_showImgOrig)

		self.label_showImgOrig.setPixmap(QPixmap.fromImage(image))
		self.scaleFactor = 1.0

		self.scrollArea_ImgOrig.setVisible(True)
		self.fitToWindowAct.setEnabled(True)
		#self.zoomInAct.setEnabled(True)
		self.updateActions()

		if not self.fitToWindowAct.isChecked():
			self.label_showImgOrig.adjustSize()

		self.label_chooseModel.setEnabled(True)
		self.comboBox_chooseModel.setEnabled(True)
		

#User View: Enable Segmentation button after choosing model		
	def choose_model_to_use(self):
		self.pushButton_segmentation.setEnabled(True)
		self.pushButton_saveImg.setEnabled(False)
		self.lineEdit_saveImg_seg.setEnabled(False) 


#User View: Segmentation
	def thread_segmentar(self):
		self.comboBox_chooseModel.setEnabled(False)
		self.pushButton_segmentation.setEnabled(False)

		selected_model=self.comboBox_chooseModel.currentText()
		filename= os.getcwd()+'/Models/'+selected_model+'.pkl'
		print("Selected model:", filename)
		model = joblib.load(filename)

		self.statusBar().showMessage("Segmenting image...")

		worker_seg = Worker(ts.segmentar, self.imgToSegment, model)

		worker_seg.signals.result.connect(self.showSegmentedImage)
		worker_seg.signals.finished.connect(self.thread_SegmentationComplete)

		#Execute
		self.threadpool.start(worker_seg)

	def thread_SegmentationComplete(self):
		print("THREAD SEGMENTATION COMPLETE!")
		
	def showSegmentedImage(self, img_segmentada):

		self.qim=qn.array2qimage(img_segmentada)

		self.label_showImgSeg = QLabel()
		self.label_showImgSeg.setBackgroundRole(QPalette.Base)
		self.label_showImgSeg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.label_showImgSeg.setScaledContents(True)
		self.scrollArea_ImgSeg.setWidget(self.label_showImgSeg)

		self.label_showImgSeg.setPixmap(QPixmap.fromImage(self.qim))
		self.label_showImgSeg.adjustSize()

		self.scaleFactor = 1.0

		self.scrollArea_ImgSeg.setVisible(True)


		self.updateActions()

		if not self.fitToWindowAct.isChecked():
			self.label_showImgOrig.adjustSize()
			self.label_showImgSeg.adjustSize()


		print("Image segmentation complete!")
		self.statusBar().showMessage("Image segmentation complete :)", 5000)
		self.lineEdit_saveImg_seg.setEnabled(True)
		self.comboBox_chooseModel.setEnabled(True)


#User View: Save Image	
	def pre_save (self):
		self.pushButton_saveImg.setEnabled(True)

	def save (self):
		input_dir = QFileDialog.getExistingDirectory(None, 'Select directory', 
			os.getcwd() + '/Saved_Segmented_Images', QFileDialog.ShowDirsOnly)
		name=self.lineEdit_saveImg_seg.text()
		w=self.qim.save(input_dir + '/' + name +'.tiff', format ='TIFF')
		if w: 
			print('Segmented image saved!')
			self.statusBar().showMessage("Segmented image saved :)", 5000)


#Build Model View: Build Model	
	def choose_model_to_use_BUILD(self, i):
		T= True
		global Params
		Params = None #se voltar a escolher um método, faz-se reset nos params
		selected_method=self.comboBox_selectMethod.currentText()
		print(selected_method)
		while T:
			if selected_method == 'Support Vector Machine':
			
				kernel= self.getKernel()
				if kernel == 'cancel':
					break
				else:
					if kernel == 'rbf':
						C=self.getC()
						if C == 'cancel':
							break
						Gamma=self.getGamma_1()
						if Gamma == 'cancel':
							break
						elif Gamma == 'float':
							Gamma=self.getGamma()
							if Gamma == 'cancel':
								break
						Params = [selected_method, kernel, C, Gamma]
					else:
						C=self.getC()
						if C == 'cancel':
							break	
						Params = [selected_method, kernel, C]
					
					print('s')
					print(Params)
					self.pushButton_chooseDataset.setEnabled(True)
					self.pushButton_chooseGT.setEnabled(True)
					self.label_importDataset.setEnabled(True)
					break
				
			else: #Forests and Trees
				
				n_estimators = self.getN_Estimators()
				if n_estimators == 'cancel':
					break
				max_depth = self.getMax_depth()
				if max_depth == 'cancel':
					break
				elif max_depth == 0:
					max_depth = None
				min_samples_split = self.getMin_samples_split()
				if min_samples_split == 'cancel':
					break
				max_leaf_nodes = self.getMax_leaf_nodes()
				if max_leaf_nodes == 'cancel':
					break
				max_features = self.getMax_features()
				if max_features == 'cancel':
					break
				if max_features == 'int':
					max_features = self.getMax_features_INT()
					if max_features == 'cancel':
						break
				Params = [selected_method, n_estimators, max_depth, min_samples_split, max_features, max_leaf_nodes]
				
				print(Params)
				self.pushButton_chooseDataset.setEnabled(True)
				self.pushButton_chooseGT.setEnabled(True)
				self.label_importDataset.setEnabled(True)
				break


#Build Model View: Define hyperparamethers SVC
	def getKernel(self):
		items = ('rbf','linear')
		item, okPressed = QInputDialog.getItem(self, "Get kernel","kernel:", items, 0, False)
		print(okPressed)
		if okPressed and item:
			return item
		else:
			T= 'cancel'
			return T

	def getC(self):
		d, okPressed = QInputDialog.getDouble(self, "Get C","Value:", 1, 0, 100, 3)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	def getGamma_1(self): 
		d, okPressed = QInputDialog.getItem(self, "Get Gamma","Value:", ['float', 'scale', 'auto'])
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	def getGamma(self): 
		d, okPressed = QInputDialog.getDouble(self, "Get Gamma","Value:", 1, 0, 100, 3)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	

#Build Model View: Define hyperparamethers Forests OR Trees
	def getN_Estimators(self): 
		d, okPressed = QInputDialog.getInt(self, "Get n_estimators","Value:", 100, 2, 100000)
		print(okPressed)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	def getMax_depth(self): 
		d, okPressed = QInputDialog.getInt(self, "Get max_depth","Value:", 2, 2, 1000000, 3)
		print(okPressed)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	def getMin_samples_split(self):
		d, okPressed = QInputDialog.getInt(self, "Get min_samples_split","Value:", 20, 2, 1000000000, 3)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
			
	def getMax_leaf_nodes(self):
		d, okPressed = QInputDialog.getInt(self, "Get max_leaf_nodes","Value:", 5, 2, 1000000000, 3)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
			
	def getMax_features(self): 
		d, okPressed = QInputDialog.getItem(self, "Get max_features","Value:", ['auto', 'log2', 'int'])
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	def getMax_features_INT(self): 
		d, okPressed = QInputDialog.getInt(self, "Get max_features","Value:", 2, 1, 3)
		if okPressed:
			print(d)
			return d
		else:
			T= 'cancel'
			return T
	
	
#Build Model View: Import dataset
	def import_dataset(self, b):
		print(b)
		f = QFileDialog.getOpenFileNames(self, "Select your "+str(b)+" dataset", 
		os.path.expanduser("~")+'/Documents', "Images (*.png *.gif *.jpg *.tif)")
		if b== 'Ground Truth':
			global filesGT #guardar os paths das imgs GT para treinar o modelo
			filesGT=f[0]
			print(filesGT)
			print(len(filesGT))
		else:
			global files_D  #guardar os paths das imgs do dataset para treinar o modelo
			files_D=f[0]
			print(files_D)
			print(len(files_D))
		if ("filesGT" in globals()) and ("files_D" in globals()):
			self.label_newModelName.setEnabled(True)
			self.lineEdit_newModelName.setEnabled(True)

#Build Model View: Write new model name
	def writeNewModelName(self):
		self.newModelName=self.lineEdit_newModelName.text()
		self.pushButton_train.setEnabled(True)
		self.pushButton_saveModel.setEnabled(False)


#Build Model View: Train model
	def thread_train(self):
		self.lineEdit_newModelName.setEnabled(False)
		self.pushButton_train.setEnabled(False)

		self.statusBar().showMessage("Training and validating model...")

		worker_train = Worker(tcm.train_model, Params, files_D, filesGT, self.newModelName)
		worker_train.signals.result.connect(self.show_TrainResults)
		worker_train.signals.finished.connect(self.thread_TrainComplete)

		self.threadpool.start(worker_train)

	def thread_TrainComplete(self):
		print("THREAD TRAIN COMPLETE!")

	def show_TrainResults(self, results):
		self.model=results.get("Model")
		time=results.get("Time")
		metrics_train=results.get("TrainMetrics")
		metrics_test=results.get("TestMetrics")
		
		print("Model trained and validated!")
		self.statusBar().showMessage("Model trained and validated :)", 5000)		

		self.label_trainMetrics.setEnabled(True)
		self.label_Acc.setEnabled(True)
		self.label_printAcc.setText("{0:.2f} %".format(100 * metrics_train.get("ACC")))
		self.label_printAcc.setEnabled(True)

		self.label_AUC.setEnabled(True)
		self.label_printAUC.setText("{0:.2f} %".format(100 * metrics_train.get("AUC")))
		self.label_printAUC.setEnabled(True)

		self.label_Precision.setEnabled(True)
		self.label_printPrecision.setText("{0:.2f} %".format(100 * metrics_train.get("Precision")))
		self.label_printPrecision.setEnabled(True)

		self.label_Recall.setEnabled(True)
		self.label_printRecall.setText("{0:.2f} %".format(100 * metrics_train.get("Recall")))
		self.label_printRecall.setEnabled(True)


		self.label_DICE.setEnabled(True)
		self.label_printDICE.setText("{0:.2f} %".format(100 * metrics_train.get("DICE")))
		self.label_printDICE.setEnabled(True)


		self.label_testMetrics.setEnabled(True)
		self.label_Acc_2.setEnabled(True)
		self.label_printAcc_2.setText("{0:.2f} %".format(100 * metrics_test.get("ACC")))
		self.label_printAcc_2.setEnabled(True)
		
		self.label_AUC_2.setEnabled(True)
		self.label_printAUC_2.setText("{0:.2f} %".format(100 * metrics_test.get("AUC")))
		self.label_printAUC_2.setEnabled(True)
		
		self.label_Precision_2.setEnabled(True)
		self.label_printPrecision_2.setText("{0:.2f} %".format(100 * metrics_test.get("Precision")))
		self.label_printPrecision_2.setEnabled(True)
		
		self.label_Recall_2.setEnabled(True)
		self.label_printRecall_2.setText("{0:.2f} %".format(100 * metrics_test.get("Recall")))
		self.label_printRecall_2.setEnabled(True)
		
		self.label_DICE_2.setEnabled(True)
		self.label_printDICE_2.setText("{0:.2f} %".format(100 * metrics_test.get("DICE")))
		self.label_printDICE_2.setEnabled(True)

		self.label_trainTime.setEnabled(True)
		self.label_printTrainTime.setText('{:.3f} s'.format(time))
		self.label_printTrainTime.setEnabled(True)

		self.pushButton_saveModel.setEnabled(True)
		self.label_ChooseImgTest.setEnabled(True)
		self.pushButton_ImportImgTrain.setEnabled(True)
		self.pushButton_ImportImgTest.setEnabled(True)
		self.pushButton_train.setEnabled(True)
		self.lineEdit_newModelName.setEnabled(True) 


#Build View: Choose image from file to test the new model
	def chooseTrainImg2testModel(self, s):
		print("click", s)
		dialog = QFileDialog(self)
		dialog.setWindowTitle("Choose an image from the imported dataset to show segmentation result")
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setDirectory(os.getcwd()+'/Segmentation_Results/Train/'+ str(self.newModelName))
		dialog.setNameFilter("Images (*.png *.gif *.jpg *.tif *.tiff)")
		dialog.setViewMode(QFileDialog.Detail)
		dialog.fileSelected.connect(self.pic_buildModel)

		dialog.show()
		dialog.exec()

	def chooseTestImg2testModel(self, s):
		print("click", s)
		dialog = QFileDialog(self)
		dialog.setWindowTitle("Choose an image from the imported dataset to show segmentation result")
		dialog.setFileMode(QFileDialog.AnyFile)
		dialog.setDirectory(os.getcwd()+'/Segmentation_Results/Test/'+ str(self.newModelName))
		dialog.setNameFilter("Images (*.png *.gif *.jpg *.tif *.tiff)")
		dialog.setViewMode(QFileDialog.Detail)
		dialog.fileSelected.connect(self.pic_buildModel)

		dialog.show()
		dialog.exec()


#Build View: Show Chosen Image to segment and test new model		
	def pic_buildModel(self, x):
		print("Selected image:", x)
		image = QImage(x)

		name_img=os.path.splitext(os.path.basename(x))[0]

		if (str(name_img[len(name_img)-6:len(name_img)+1]))=='_train':
			name_selected=str(name_img[0:len(name_img)-6:])
		elif (str(name_img[len(name_img)-5:len(name_img)+1]))=='_test':
			name_selected=str(name_img[0:len(name_img)-5:])
		else:
			print("Something's wrong in the selected image")
	
		for i in range(len(files_D)):
			possible_selected=os.path.splitext(os.path.basename(files_D[i]))[0]
			if possible_selected== name_selected:
				index_gt=i
				break

		path_gt=filesGT[i]


		image_gt=QImage(path_gt)

		self.label_showImgGT = QLabel()
		self.label_showImgGT.setBackgroundRole(QPalette.Base)
		self.label_showImgGT.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.label_showImgGT.setScaledContents(True)
		self.scrollArea_GT.setWidget(self.label_showImgGT)

		self.label_showImgGT.setPixmap(QPixmap.fromImage(image_gt))

		self.label_showImgSegTest = QLabel()
		self.label_showImgSegTest.setBackgroundRole(QPalette.Base)
		self.label_showImgSegTest.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.label_showImgSegTest.setScaledContents(True)
		self.scrollArea_ImgSegBuild.setWidget(self.label_showImgSegTest)

		self.label_showImgSegTest.setPixmap(QPixmap.fromImage(image))
		self.scaleFactor = 1.0

		self.scrollArea_GT.setVisible(True)
		self.scrollArea_ImgSegBuild.setVisible(True)
		self.fitToWindowAct.setEnabled(True)
		self.updateActions()

		if not self.fitToWindowAct.isChecked():
			self.label_showImgGT.adjustSize()
			self.label_showImgSegTest.adjustSize()


#Build View: Save Model
	def saveNewModel(self):
		input_dir = QFileDialog.getExistingDirectory(None, 'Select a directory to save model', 
			os.getcwd()+'/Models', QFileDialog.ShowDirsOnly)
		
		#If cancel is selected an exception occurs
		try:
			filename=input_dir+'/'+self.newModelName+'.pkl'
			joblib.dump(self.model, filename)

			print('Model saved in path: ', filename)
			self.statusBar().showMessage('Model saved :)', 5000)
			self.updateComboBoxChooseModel()
			index=int(self.comboBox_chooseModel.findText(str(self.newModelName)))
			print('Index of created model in comboBox_chooseModel: ', index)
			self.comboBox_chooseModel.setCurrentIndex(index)

		except:
			print("An exception occurred")



#Menu Zoom		
	def zoomIn(self):
		self.scaleImage(1.25)

	def zoomOut(self):
		self.scaleImage(0.75)

	def normalSize(self):
		if self.tabWidget.currentIndex() == self.tabWidget.indexOf(self.tab_useModel):
			if hasattr(self, 'label_showImgOrig'):
				self.label_showImgOrig.adjustSize()
			if hasattr(self, 'label_showImgSeg'):
				self.label_showImgSeg.adjustSize()
		elif self.tabWidget.currentIndex() == self.tabWidget.indexOf(self.tab_buildModel): 
			if hasattr(self, 'label_showImgGT'):
				self.label_showImgGT.adjustSize()
			if hasattr(self, 'label_showImgSegTest'):
				self.label_showImgSegTest.adjustSize()

		self.scaleFactor = 1.0

	def fitToWindow(self):
		fitToWindow = self.fitToWindowAct.isChecked()
		print('fitToWindow:', fitToWindow)
		self.scrollArea_ImgOrig.setWidgetResizable(fitToWindow)
		self.scrollArea_ImgSeg.setWidgetResizable(fitToWindow)
		self.scrollArea_GT.setWidgetResizable(fitToWindow)
		self.scrollArea_ImgSegBuild.setWidgetResizable(fitToWindow)

		if not fitToWindow:
			self.normalSize()

		self.updateActions()
		
	def updateActions(self):
		self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
		self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
		self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

	def scaleImage(self, factor):
		self.scaleFactor *= factor
		if self.tabWidget.currentIndex() == self.tabWidget.indexOf(self.tab_useModel):
			if hasattr(self, 'label_showImgOrig'):
				self.label_showImgOrig.resize(self.scaleFactor * self.label_showImgOrig.pixmap().size())
			if hasattr(self, 'label_showImgSeg'):
				self.label_showImgSeg.resize(self.scaleFactor * self.label_showImgSeg.pixmap().size())
			self.adjustScrollBar(self.scrollArea_ImgOrig.horizontalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_ImgOrig.verticalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_ImgSeg.horizontalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_ImgSeg.verticalScrollBar(), factor)

		elif self.tabWidget.currentIndex() == self.tabWidget.indexOf(self.tab_buildModel):
			if hasattr(self, 'label_showImgGT'):
				self.label_showImgGT.resize(self.scaleFactor * self.label_showImgGT.pixmap().size())
			if hasattr(self, 'label_showImgSegTest'):
				self.label_showImgSegTest.resize(self.scaleFactor * self.label_showImgSegTest.pixmap().size())
			self.adjustScrollBar(self.scrollArea_GT.horizontalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_GT.verticalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_ImgSegBuild.horizontalScrollBar(), factor)
			self.adjustScrollBar(self.scrollArea_ImgSegBuild.verticalScrollBar(), factor)


		self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
		self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

#Menu bar: Create Actions
	def createActions(self):
		self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
		self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
		self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
		self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F", triggered=self.fitToWindow)
		self.tableParamethers = QAction("&Model Paramethers", self, shortcut="Ctrl+t", enabled=True, triggered=self.tableModelHyperparamethers)
		self.aboutAct = QAction("&About", self, triggered=self.about)
		self.resetAct = QAction("&RESET", self, enabled=True, triggered=self.reset)
		self.exit = QAction("&Exit", self, shortcut="Ctrl+Q", enabled=True, triggered=self.close) 

#Menu bar: Create Menus
	def createMenus(self):

		self.viewMenu = QMenu("&Zoom", self)
		self.viewMenu.addAction(self.zoomInAct)
		self.viewMenu.addAction(self.zoomOutAct)
		self.viewMenu.addAction(self.normalSizeAct)
		self.viewMenu.addSeparator()
		self.viewMenu.addAction(self.fitToWindowAct)

		self.menuBar().addMenu(self.viewMenu)

		self.menuRetina.addAction(self.tableParamethers)
		self.menuRetina.addAction(self.aboutAct)
		self.menuRetina.addAction(self.resetAct)
		self.menuRetina.addAction(self.exit) 
		
	def adjustScrollBar(self, scrollBar, factor):
		scrollBar.setValue(int(factor * scrollBar.value()
		+ ((factor - 1) * scrollBar.pageStep() / 2)))

#Menu bar: Create model Paramethers
	def tableModelHyperparamethers(self):
		ex = MyMessageBox ()

#Menu bar: Create about
	def about(self):
		QMessageBox.about(self, "About RetinaPlus",
		                  "<p>The <b>RetinaPlus</b> app permits the user to segment"
		                  "retina images throw a ML model already created and saved."
		                  " This action is offered in the UseModel Tab."
		                  "<p>The BuildModel Tabe allows the construction, training "
		                  "and testing of the new model chosen by the user himself. "
		                  "Test and Train metrics are shown to evalute the model. "
		                  "<p>The app also provides other funcionalities, such as displaying"
		                  " the segmented images and saving them, as well as the model created.")

#Menu bar: Create reset
	def reset(self):
		defaul_models=['Extremely_Randomized_Trees.pkl', 'Random_Decision_Forests.pkl',
						'SVM_Linear.pkl', 'SVM_RBF.pkl']
		d = os.getcwd()
		for name in os.listdir(d): #lista tudo o que esta em d
			path = os.path.join(d, name)
			if os.path.isdir(path): #so as que sao diretorias (para nao apanhar os ficheiros de codigo da app)
				for file in os.listdir(path): #tudo o que esta dentro da diretoria
					if file not in defaul_models:
						file_path = os.path.join(path, file) 
						try:
							if os.path.isfile(file_path) or os.path.islink(file_path):
								os.unlink(file_path)
							elif os.path.isdir(file_path):
								shutil.rmtree(file_path)
						except Exception as e:
							print('Failed to delete %s. Reason: %s' % (file_path, e))
				if path.endswith('Segmentation_Results'):
					os.mkdir(path+'/Train')
					os.mkdir(path+'/Test')

		self.initialSavedModels = self.getModelList()
		self.initialNrSavedModels=len(self.initialSavedModels)

		self.tabWidget.removeTab(2)
		self.tabWidget.removeTab(1)
		self.comboBox.setCurrentIndex(-1)


if __name__ == '__main__':

	app = QApplication(sys.argv)

	window = MainWindow()
	window.show()

	app.exec_()

	



