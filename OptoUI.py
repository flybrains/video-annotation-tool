
import sys
import cv2
import time
import os
from os import system
import numpy as np
import pickle
import serial
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLabel, QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QPixmap

import moviepy
from moviepy.editor import ImageSequenceClip

import flycapture2 as fc2

cwd = os.getcwd()
qtCreatorFile = cwd+"/cameraUI.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class ErrorMsg(QtWidgets.QMessageBox):
	def __init__(self, msg, parent=None):
		super(ErrorMsg, self).__init__(parent)
		self.setIcon(QtWidgets.QMessageBox.Critical)
		self.setText(msg)
		self.setWindowTitle('Error')

class WarningMsg(QtWidgets.QMessageBox):
	def __init__(self, msg, parent=None):
		super(WarningMsg, self).__init__(parent)
		self.setText(msg)
		self.setWindowTitle('Warning')

class Block(object):
	def __init__(self, duration, lightColor, lightIntensity, recording, lightDspTxt, recString):
		self.duration = duration
		self.lightColor = lightColor
		self.lightIntensity = lightIntensity
		self.recording = recording
		self.lightDspTxt = lightDspTxt
		self.recString = recString

class CameraThread(QThread):
	changePixmap = pyqtSignal(QImage)
	finished = pyqtSignal()
	count = pyqtSignal(int, name='count')

	def __init__(self, nFrames, saveDir, write=False, testMode=False):
		QThread.__init__(self)
		self.nFrames = nFrames
		self.saveDir = saveDir
		self.write = write
		self.testMode = testMode

	def __del__(self):
		self.wait()

	def run(self):
		self.threadactive=True


		if self.testMode==True:
			cap = cv2.VideoCapture('/home/patrick/Desktop/out16.mp4')
		else:
			try:
				cap = fc2.Context()
				cap.connect(*cap.get_camera_from_index(0))
				cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_30)
				m, f = cap.get_video_mode_and_frame_rate()
				p = cap.get_property(fc2.FRAME_RATE)
				cap.set_property(**p)
				cap.start_capture()
			except fc2.ApiError:
				msg = 'Make sure a camera is connected  '
				self.error = ErrorMsg(msg)
				self.error.show()
				self.threadactive=False
				self.finished.emit()
				return None


		i = 0

		while (i < self.nFrames and self.threadactive):

			if self.testMode==True:
				ret, frame = cap.read()
				frame = cv2.resize(frame, (420,420))
				frame = frame.copy()
			else:
				img = fc2.Image()
				cap.retrieve_buffer(img)
				frame = np.array(img)
				frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

			h, w, ch = frame.shape
			bytesPerLine = ch * w
			convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
			p = convertToQtFormat.scaled(420, 420, QtCore.Qt.KeepAspectRatio)
			self.changePixmap.emit(p)
			if self.write==True:
				j = str(i)
				k = str(j.zfill(6))
				cv2.imwrite(self.saveDir+'/{}.jpg'.format(k), frame)
			i = i+1
			prog = int(100*(i/self.nFrames))
			if self.write==True:
				self.count.emit(prog)
		self.finished.emit()

		if self.testMode==True:
			cap.release()

		else:
			cap.stop_capture()
			cap.disconnect()

	def stop(self):
		self.threadactive=False
		self.finished.emit()
		self.wait()

class LightThread(QThread):
	lightsFinished = pyqtSignal()

	def __init__(self, ser, programLists):
		QThread.__init__(self)
		self.ser = ser
		self.programLists = programLists

	def __del__(self):
		self.wait()

	def run(self):

		self.threadactive=True

		timeList = self.programLists[0]
		cfgList = self.programLists[1]
		endTarget = '10'+'\n'


		for index, item in enumerate(cfgList):
			state = int(item[0])
			intensity = int(item[1])
			if intensity ==100:
				intensity = 99

			target = str(state)+str(intensity)


			sendStr = str(target)+'\n'

			dur = timeList[index]

			t0 = time.time()

			while (time.time() - t0) < dur:
				self.ser.write(str.encode(sendStr))
				time.sleep(0.1)

		self.ser.write(str.encode(endTarget))
		self.lightsFinished.emit()

	def stop(self):
		self.threadactive=False
		self.lightsFinished.emit()
		self.wait()


class LiveImage(QMainWindow):
	def __init__(self, parent=None):
		super(LiveImage, self).__init__(parent)

class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self):
		# General Initialization
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.title = 'Behavior Experiment Controller'
		self.setWindowTitle(self.title)
		self.setFixedSize(self.size())

		# Camera Thread Business
		self.startCamPushButton.clicked.connect(self.runCam)
		self.stopCamPushButton.clicked.connect(self.stopCam)

		# Button Connections
		self.addBlockPB.clicked.connect(self.addBlock)
		self.addDupBlocksPB.clicked.connect(self.addDupBlocks)
		self.runPB.clicked.connect(self.runExperiment)
		self.deleteBlockPB.clicked.connect(self.deleteBlock)
		self.pickSavePushButton.clicked.connect(self.pickSaveFolder)
		self.saveProgramPB.clicked.connect(self.saveProgram)
		self.loadProgramPB.clicked.connect(self.loadProgram)
		self.blockList = []
		dString = "{},\t{}\t{}\t{}".format('#', 'Dur (s)', 'Color', 'Intensity')
		lString = "--------------------------------------------------------------------"
		self.programList.addItems([dString, lString])
		self.arduinoCommText.setText('/dev/ttyACM0')
		self.arduinoBaudText.setText('9600')
		self.progressBar.hide()
		self.progressBar.setValue(0)

		self.viewerLockout = False
		self.setBG()


	def updatePGB(self, valueProg):
		#print(valueProg)
		self.progressBar.setValue(valueProg)
		self.progressBar.show()

	def addBlock(self):
		if self.addGreenRadioButton.isChecked():
			lightDspTxt = "Blue"
			lightColor = 4
		if self.addGreenRadioButton.isChecked():
			lightDspTxt = "Green"
			lightColor = 3
		elif self.addRedRadioButton.isChecked():
			lightDspTxt = "Red"
			lightColor = 2
		else:
			lightDspTxt = "No Light"
			lightColor = 1

		duration = float(self.addTimeSpinBox.value())

		if lightColor == 0:
			lightIntensity = 0
		else:
			lightIntensity = int(self.intensitySpinBox.value())

		recording = True
		recString = 'ON'

		if (lightIntensity == 0) and (lightColor is not 1):
			lightColor = 1
			lightDspTxt = 'No Light'

		newBlock = Block(duration, lightColor, lightIntensity, recording, lightDspTxt, recString)
		self.blockList.append(newBlock)
		listPos = len(self.blockList)

		if lightColor==1:
			lightColor='No Light'
		elif lightColor==2:
			lightColor = 'Red'
		elif lightColor==3:
			lightColor = 'Green'
		else:
			lightColor = 'Blue'

		dispString = "{},\t{}\t{}\t{}".format(listPos, str(duration), lightColor, lightIntensity)
		self.programList.addItems([dispString])
		return None

	def deleteBlock(self):
		d = self.programList.currentRow() - 2
		item = self.programList.takeItem(self.programList.currentRow())
		item = None
		del self.blockList[d]
		entries = [self.programList.item(i).text() for i in range(self.programList.count())]
		reindexed = []
		for idx, entry in enumerate(entries[2:]):
			listOfInfos = entry.split(",")
			listOfInfos[0] = idx+1
			reconstructed = '{},{}'.format(listOfInfos[0], listOfInfos[1])
			reindexed.append(reconstructed)
		self.programList.clear()
		dString = "{},\t{}\t{}\t{}".format('#', 'Dur (s)', 'Color', 'Intensity')
		lString = "--------------------------------------------------------------------"
		self.programList.addItems([dString, lString])
		for reindex in reindexed:
			self.programList.addItems([reindex])
		return None

	def saveProgram(self):
		self.programSavePath = QFileDialog.getSaveFileName(self, 'Select Save Directory', os.getcwd())

		self.programSavePath = self.programSavePath[0]+".pkl"
		entries = [self.programList.item(i).text() for i in range(self.programList.count())]
		savePack = {'dispList':entries,
					'blockList':self.blockList}
		pickle_out = open(self.programSavePath,"wb")
		pickle.dump(savePack, pickle_out)
		pickle_out.close()
		return None

	def loadProgram(self):
		fname = QFileDialog.getOpenFileName(self, 'Select Program to Open', os.getcwd())
		self.openProgramPath = str(fname[0])
		pickle_in = open(self.openProgramPath, "rb")
		savePack = pickle.load(pickle_in)
		self.programList.clear()
		self.blockList = savePack['blockList']
		for entry in savePack['dispList']:
			self.programList.addItems([entry])
		return None

	def addDupBlocks(self):
		single = False
		multi = False


		if self.dupBlockText.toPlainText() != '':
			single = True
			idx = int(self.dupBlockText.toPlainText()) -1
			entries = [self.programList.item(i).text() for i in range(self.programList.count())]
			toAdd = entries[idx+2]

			blockToReplicate = self.blockList[idx]
			blockToAdd = Block(blockToReplicate.duration, blockToReplicate.lightColor,
								   blockToReplicate.lightIntensity, blockToReplicate.recording,
								   blockToReplicate.lightDspTxt,blockToReplicate.recString)
			self.blockList.append(blockToAdd)

			self.dupBlockText.clear()
			listOfInfos = toAdd.split(",")
			listOfInfos[0] = str(len(self.blockList))
			reconstructed = '{},{}'.format(listOfInfos[0], listOfInfos[1])
			self.programList.addItems([reconstructed])

		else:
			if (self.dupBlocksFirstText.toPlainText() != "") and (self.dupBlocksLastText.toPlainText() != ""):
				if single == True:

					msg = 'Cannot add single block and range in same operation'
					self.warning = WarningMsg(msg)
					self.warning.show()

					self.dupBlocksFirstText.clear()
					self.dupBlocksLastText.clear()
				else:
					multi = True
					idxLo = int(self.dupBlocksFirstText.toPlainText())
					idxHi = int(self.dupBlocksLastText.toPlainText())

					if idxLo >= idxHi:
						msg = 'Last entry in range must be larger than first'
						self.error = ErrorMsg(msg)
						self.error.show()


					copyBlocks = list(np.arange(idxLo, (idxHi+1), 1))


					for i in range(self.programList.count()):
						if i in copyBlocks:

							blockToReplicate = self.blockList[i-1]
							blockToAdd = Block(blockToReplicate.duration, blockToReplicate.lightColor,
												   blockToReplicate.lightIntensity, blockToReplicate.recording,
												   blockToReplicate.lightDspTxt,blockToReplicate.recString)
							self.blockList.append(blockToAdd)

							index = str(len(self.blockList))
							toAdd = self.programList.item(i+1).text()
							listOfInfos = toAdd.split(",")
							listOfInfos[0] = index
							reconstructed = '{},{}'.format(listOfInfos[0], listOfInfos[1])
							self.programList.addItems([reconstructed])
						else:
							pass
				self.dupBlocksFirstText.clear()
				self.dupBlocksLastText.clear()

			return None

	def pickSaveFolder(self):
		fname = QFileDialog.getExistingDirectory(self, 'Select Save Directory')
		self.savePath = str(fname)
		self.savePathLabel.setText(self.savePath)
		return None

	@pyqtSlot(QImage)
	def setImage(self, image):
		self.label.setPixmap(QPixmap.fromImage(image))
		self.label.show()

	def runCam(self):
		if self.viewerLockout==False:
			self.setWindowTitle(self.title)
			self.label = QLabel(self)
			self.label.move(10, 40)
			self.label.refsize(420, 420)
			self.camThread = CameraThread(100000, None, write=False)
			self.camThread.changePixmap.connect(self.setImage)
			self.camThread.count.connect(self.updatePGB)
			self.camThread.start()
			self.camThread.finished.connect(self.setBG)
			self.viewerLockout = True
		else:
			msg = 'Cannot start new viewing window when one is active  '
			self.warning = WarningMsg(msg)
			self.warning.show()
			return None

	def stopCam(self):
		self.label.hide()
		self.camThread.stop()
		self.camThread.quit()
		self.camThread.wait()
		self.viewerLockout = False

	def setBG(self):
		try:
			self.label.hide()
		except AttributeError:
			pass
		self.l1 = QLabel(self)
		self.l1.resize(400, 120)
		self.l1.move(45,200)
		self.l1.setText('No Video Feed Connected')
		self.l1.setFont(QtGui.QFont('SansSerif', 20))
		self.l1.show()

	def serialCleanup(self):
		sendStr = '10'+'\n'
		self.ser.write(str.encode(sendStr))
		if self.saveVideo:
			clip = ImageSequenceClip(self.savePath +"/"+self.datetimeString, fps=30)


			clip.write_videofile(self.savePath+'/videos/'+self.datetimeString+'.mp4', audio=False)
			#clip.write_videofile(self.savePath+'/videos/'+self.datetimeString+'.avi', audio=False, codec='rawvideo')

	def runExperiment(self):

		self.saveVideo = bool(self.saveVideoCheckBox.isChecked())

		if (self.arduinoCommText.toPlainText() is ""):
			msg = 'Must specify Arduino COMM port  '
			self.error = ErrorMsg(msg)
			self.error.show()
			return None

		if (self.arduinoBaudText.toPlainText() is ''):
			msg = 'Must specify Arduino Baudrate  '
			self.error = ErrorMsg(msg)
			self.error.show()
			return None

		try:
			self.savePath
		except AttributeError:
			msg = 'Must select unique save location '
			self.error = ErrorMsg(msg)
			self.error.show()
			return None



		self.progressBar.show()
		self.setWindowTitle(self.title)
		self.label = QLabel(self)
		self.label.move(10, 40)
		self.label.resize(420, 420)

		self.comm = str(self.arduinoCommText.toPlainText())
		self.baud = int(self.arduinoBaudText.toPlainText())

		try:
			self.ser = serial.Serial(self.comm, self.baud)
		except serial.serialutil.SerialException:
			msg = 'Unable to establish connection with Arduino. Check COMM and Baud and connection with board. Re-upload program if necessary '
			self.error = ErrorMsg(msg)
			self.error.show()
			return None

		cap = fc2.Context()
		cap.connect(*cap.get_camera_from_index(0))
		cap.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2.FRAMERATE_30)
		m, f = cap.get_video_mode_and_frame_rate()
		p = cap.get_property(fc2.FRAME_RATE)
		cap.set_property(**p)
		cap.start_capture()


		time.sleep(1)

		dt = datetime.now()
		self.datetimeString = str(dt.month)+"_"+str(dt.day)+"_"+str(dt.year)+"_"+str(dt.hour)+str(dt.minute)

		timeList = [float(block.duration) for block in self.blockList]
		cfgList = [[block.lightColor, block.lightIntensity] for block in self.blockList]
		self.programLists = [timeList, cfgList]
		timeSum = np.sum(timeList)
		nFrames = int(timeSum*30)

		outFolder = self.savePath + "/" + self.datetimeString

		if outFolder in os.listdir():
			outFolder = outFolder + '_a'
		else:
			os.mkdir(outFolder)

		if 'videos' in os.listdir(self.savePath):
			pass
		else:
			os.mkdir(self.savePath + '/videos')

		try:
			self.camThread = CameraThread(nFrames, outFolder, write=True)
			self.camThread.changePixmap.connect(self.setImage)
			self.camThread.count.connect(self.updatePGB)
			self.camThread.start()
			self.camThread.finished.connect(self.setBG)
		except fc2.ApiError:
			msg = 'Make sure a camera is connected  '
			self.error = ErrorMsg(msg)
			self.error.show()
			return None

		self.lightThread = LightThread(self.ser, self.programLists)
		self.lightThread.start()
		self.lightThread.lightsFinished.connect(self.serialCleanup)




if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())
