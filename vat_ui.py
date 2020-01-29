import sys
import os
import numpy as np
import cv2
import threading
import vat_core as vc
import vat_utilities as vu
import pickle
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QSpacerItem, QProgressDialog, QDialog, QWidget,QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QMenu, QAction, QSpinBox, QGridLayout, QFileDialog


class ExperimentConfigWindow(QDialog):
	got_values = pyqtSignal(int)
	start_bg_comp = pyqtSignal(str, int)

	def __init__(self,parent=None):
		super(ExperimentConfigWindow, self).__init__(parent)

		self.pb1 = QPushButton("Select Video")
		self.pb2 = QPushButton("Define Chamber")
		self.l3 = QLabel('N Food Patch')
		self.sp3 = QSpinBox()
		self.pb4 = QPushButton("Define Food Patches")
		self.pb5 = QPushButton("Set Thresholds")

		self.l7 = QLabel('N Animals')
		self.sp7 = QSpinBox()
		self.l8 = QLabel('N Frames')
		self.sp8 = QSpinBox()

		self.savepb = QPushButton('Apply Configuration')

		self.indicator_label1 = QLabel()
		self.indicator_label2 = QLabel()
		self.indicator_label3 = QLabel()
		self.indicator_label4 = QLabel()
		self.indicator_label5 = QLabel()
		self.indicator_label1.resize(24,24)
		self.indicator_label2.resize(24,24)
		self.indicator_label3.resize(24,24)
		self.indicator_label4.resize(24,24)
		self.indicator_label5.resize(24,24)
		self.indicator_label1.setMaximumWidth(24)
		self.indicator_label2.setMaximumWidth(24)
		self.indicator_label3.setMaximumWidth(24)
		self.indicator_label4.setMaximumWidth(24)
		self.indicator_label5.setMaximumWidth(24)
		self.indicator_label1.setStyleSheet('background-color: red')
		self.indicator_label2.setStyleSheet('background-color: red')
		self.indicator_label3.setStyleSheet('background-color: red')
		self.indicator_label4.setStyleSheet('background-color: red')
		self.indicator_label5.setStyleSheet('background-color: red')

		self.grid = QGridLayout()
		self.grid.setSpacing(5)
		self.grid.addWidget(self.indicator_label1,1,0)
		self.grid.addWidget(self.pb1, 1, 1)
		self.grid.addWidget(self.indicator_label2,2,0)
		self.grid.addWidget(self.pb2, 2, 1)
		self.grid.addWidget(self.l3, 3, 0)
		self.grid.addWidget(self.sp3, 3, 1)
		self.grid.addWidget(self.indicator_label3,4,0)
		self.grid.addWidget(self.pb4, 4, 1)
		self.grid.addWidget(self.indicator_label4,5,0)
		self.grid.addWidget(self.pb5, 5, 1)
		# self.grid.addWidget(self.indicator_label5,6,0)
		# self.grid.addWidget(self.pb6, 6, 1)
		self.grid.addWidget(self.l7, 7, 0)
		self.grid.addWidget(self.l8, 8, 0)
		self.grid.addWidget(self.sp7, 7, 1)
		self.grid.addWidget(self.sp8, 8, 1)
		self.grid.addWidget(self.savepb, 9, 1)
		self.setLayout(self.grid)
		self.resize(300, 300)
		self.setFixedSize(self.size())
		self.setWindowTitle('Experiment Configuration')
		self.savepb.clicked.connect(self.store_values)

		self.pb1.clicked.connect(self.select_video)
		self.pb2.clicked.connect(self.define_bowl_mask)
		self.pb4.clicked.connect(self.define_food_patch)
		self.pb5.clicked.connect(self.define_thresholds)

	@pyqtSlot(np.ndarray)
	def on_finished(self, array):
		self.bg_progress_window.close()
		self.bg = array

	def select_video(self):
		self.video_address = QFileDialog.getOpenFileName(self, 'Select Video to Analyze', os.getcwd())[0]
		self.indicator_label1.setStyleSheet('background-color: green')
		return None

	def define_bowl_mask(self):
		ed = vu.EllipseDrawer(self.video_address)
		ed.define_bowl_mask()
		self.bowl_mask, self.mask_centroid = ed.get_bowl_mask()
		self.indicator_label2.setStyleSheet('background-color: green')

	def define_food_patch(self, patches):
		ed = vu.EllipseDrawer(self.video_address)
		self.n_food_patch = int(self.sp3.value())
		if self.n_food_patch==0:
			self.food_patch_mask=None
		else:
			ed.define_food_patches(self.n_food_patch)
			self.food_patch_mask = ed.get_food_patches()
		self.indicator_label3.setStyleSheet('background-color: green')

		self.bg_progress_window = QProgressDialog(self)
		self.bg_progress_window.hide()
		self.bg_progress_window.setLabelText('Computing Average Background Features')
		thread = QThread(self)
		thread.start()
		self.bgc = BGCalculator()
		self.start_bg_comp.connect(self.bgc.calculateBGwithProgress)
		self.bgc.progressChanged.connect(self.bg_progress_window.setValue)
		self.bgc.finished.connect(self.on_finished)
		self.bgc.started.connect(self.bg_progress_window.show)
		self.bgc.moveToThread(thread)
		self.start_bg_comp.emit(self.video_address, self.mask_centroid[0])

	def define_thresholds(self):
		thresholder = vu.Thresholder(self.video_address, self.bg, self.mask_centroid[0], self.bowl_mask)
		self.thresh, self.small, self.large, self.solidity, self.extent, self.aspect, self.arc = thresholder.get_values()
		self.indicator_label4.setStyleSheet('background-color: green')


	def store_values(self):
		self.n_patches = int(self.sp3.value())
		self.n_animals = int(self.sp7.value())
		self.n_frames = int(self.sp8.value())
		self.got_values.emit(1)
		self.close()
		return None

	def get_values(self):
		return self.thresh, self.large, self.small,self.solidity, self.extent, self.aspect, self.arc, self.bg, self.video_address, self.n_patches, self.n_animals, self.n_frames, self.bowl_mask, self.food_patch_mask, self.mask_centroid

class BGCalculator(QObject):
	progressChanged = pyqtSignal(int)
	started = pyqtSignal()
	finished = pyqtSignal(np.ndarray)

	def calculateAndUpdate(self, done, total):
		progress = int(round((done / float(total)) * 100))
		self.progressChanged.emit(progress)

	@pyqtSlot(str, int)
	def calculateBGwithProgress(self, video_address, center):

		cap = cv2.VideoCapture(video_address)
		_, frame =cap.read()
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		height = frame.shape[0]
		xl, xr = center-int(height/2), center+int(height/2)

		depth = int(np.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT)/300))
		blank = []
		frameIdxs = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-100), num=depth)
		frameIdxs = [int(e) for e in frameIdxs]

		self.started.emit()

		for idx,i in enumerate(frameIdxs):
			cap.set(cv2.CAP_PROP_POS_FRAMES, i)
			_, frame = cap.read()
			frame = frame[:,xl:xr,:]
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			self.calculateAndUpdate(idx,len(frameIdxs))
			blank.append(frame[:,:])
		blank = np.asarray(blank)
		bg = np.mean(blank, axis=0)
		bg = bg.astype(np.uint8)

		self.finished.emit(bg)

class MainWindow(QMainWindow):
	frameTuple = pyqtSignal(int, int)
	update_combos = pyqtSignal()
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.experiment_configurator = ExperimentConfigWindow()
		self.tracking_fixer = FixTrackingWindow()
		self.behavior_selector_widget = BehaviorSelectorWidget(self)
		self.image_widget = ImageWidget(self)
		_widget = QWidget()
		_layout = QHBoxLayout(_widget)
		_layout.addWidget(self.image_widget)
		_layout.addWidget(self.behavior_selector_widget)
		self.setCentralWidget(_widget)
		newAct = QAction('New Analysis', self)
		newAct.triggered.connect(self.new_analysis_cascade)
		self.experiment_configurator.got_values.connect(self.load_info)
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('File')
		fileMenu.addAction(newAct)
		self.behavior_selector_widget.toggleTrack.connect(self.toggle_image)
		self.behavior_selector_widget.newListofBehaviors.connect(self.update_list_of_behaviors)
		self.behavior_selector_widget.fix_now.connect(self.fix_tracking)
		self.behavior_selector_widget.signal_save.connect(self.save_to_cache)
		self.tracking_fixer.split_read.connect(self.split_read)
		self.tracking_fixer.remove_read.connect(self.remove_read)
		self.tracking_fixer.add_read.connect(self.add_read)

		self.image_widget.request_frame_status.connect(self.send_frame_status)
		self.currentFrame = 1
		self.n_frames = 0
		self.frameLabel = '{} / {}'.format(self.currentFrame+1, self.n_frames)
		self.mainWidgetShowsTrack = True
		self.colorMap = [(153,255,153),(204,255,153),(255,153,255),(204,153,255),(51,51,255),(51,153,255),(51,255,255),(51,255,153),(255,51,255),(153,51,255),(153,153,255),(153,204,255),(255,255,153),(255,204,153),(255,153,204),(51,255,51),(153,255,51),(255,255,51),(255,153,51),(255,51,51),(255,51,153),(153,255,255),(153,255,204),(0,0,204),(0,204,102),(204,204,0),(204,102,0),(153,153,0),(0,204,0),(153,0,153)]

	@pyqtSlot()
	def save_to_cache(self):
		with open(os.path.join(os.getcwd(), 'experiment_cache','cache.pkl'), 'wb') as f:
			pickle.dump(self.video_information, f)

	@pyqtSlot(str)
	def send_frame_status(self, id):
		if id=='inc':
			self.currentFrame += 1
			if self.currentFrame > self.n_frames:
				self.currentFrame = self.n_frames
		elif id == 'dec':
			self.currentFrame -= 1
			if self.currentFrame < 1:
				self.currentFrame= 1
		else:
			pass

		self.currentFrameIndex = self.frameIdxs[self.currentFrame-1]
		if self.currentFrame==self.n_frames:
			self.currentFrameIndex = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)-6)


		self.frameTuple.emit(self.currentFrame, self.n_frames)
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrameIndex)
		self.update_raw_image(True)
		self.mainWidgetShowsTrack=True

	def track_raw_image(self):

		def _get(cnt):
			x,y,w,h = cv2.boundingRect(cnt)
			area = cv2.contourArea(cnt)
			rect_area = w*h
			extent = float(area)/rect_area
			aspect_ratio = float(w)/h
			hull = cv2.convexHull(cnt)
			hull_area = cv2.contourArea(hull)
			if hull_area >0:
				solidity = float(area)/hull_area
			else:
				solidity=100000
			if area==0:
				arc = 1000
			else:
				arc = (cv2.arcLength(cnt,True)/cv2.contourArea(cnt))
			return aspect_ratio, extent, solidity, cv2.contourArea(cnt), arc

		if self.active_frame_info.tracked==False:
			self.tracked_image = self.currentRawImage.copy()
			self.tracked_image = cv2.cvtColor(self.tracked_image, cv2.COLOR_BGR2GRAY)
			self.tracked_image=self.tracked_image-self.bg + 10
			height = self.currentRawImage.shape[0]
			self.bm = cv2.cvtColor(self.bowl_mask[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:], cv2.COLOR_BGR2GRAY)
			self.tracked_image = cv2.bitwise_and(self.tracked_image, self.bm)
			_,self.threshed_frame = cv2.threshold(self.tracked_image,self.thresh,255,cv2.THRESH_BINARY)
			self.imagem = cv2.bitwise_not(self.threshed_frame)
			self.contours, hier = cv2.findContours(self.imagem, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			self.valids = [cnt for cnt in self.contours if (_get(cnt)[3] > self.small) and (_get(cnt)[3] < self.large) and _get(cnt)[0] < self.aspect and _get(cnt)[1] < self.extent and _get(cnt)[2] > self.solidity and _get(cnt)[-1]<self.arc]
			self.display_frame = self.currentRawImage
			for idx, c in enumerate(self.valids):
				M = cv2.moments(c)
				if M['m00'] != 0:
					cx = np.float16(M['m10']/M['m00'])
					cy = np.float16(M['m01']/M['m00'])
					contour_pt = vc.ContourPoint(idx+1, cx, cy, c, 1)
					self.active_frame_info.add_contour_point(contour_pt)
					self.display_frame = cv2.drawContours(self.display_frame, [self.valids[idx]], -1, self.colorMap[idx], 1)

					self.display_frame = cv2.putText(self.display_frame,"{}".format(contour_pt.id),(int(cx+10),int(cy+10)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)
			self.active_frame_info.tracked = True
		else:
			self.display_frame = self.currentRawImage
			for idx, c in enumerate(self.active_frame_info.get_list_of_contour_points()):
				if c.label_config==1:
					xp, yp = 10,10
				elif c.label_config==2:
					xp, yp = -10,-10
				elif c.label_config==3:
					xp, yp = 10,-10
				elif c.label_config==4:
					xp, yp = -10,10
				if c.c is not None:
					self.display_frame = cv2.drawContours(self.display_frame, [c.c], -1, self.colorMap[idx], 1)
					self.display_frame = cv2.putText(self.display_frame,"{}".format(c.id),(int(c.x+xp),int(c.y+yp)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)
				else:
					self.display_frame = cv2.circle(self.display_frame,(c.x,c.y),4,self.colorMap[idx],1)
					self.display_frame = cv2.putText(self.display_frame,"{}".format(c.id),(int(c.x+xp),int(c.y+yp)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)


	@pyqtSlot()
	def update_list_of_behaviors(self):
		self.active_frame_info.saved = True
		self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: green')

		self.active_frame_info.behavior_list = self.behavior_selector_widget.list_of_behaviors
		self.active_frame_info.position_list = [[e.id, e.x, e.y] for e in self.active_frame_info.list_of_contour_points]


	def update_raw_image(self, tracking):
		self.active_frame_info = self.video_information.get_frame_list()[self.currentFrame-1]

		if self.active_frame_info.saved:
			self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: green')
		else:
			self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: red')


		self.update_combos.emit()


		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrameIndex)
		_, self.currentRawImage = self.cap.read()
		height = self.currentRawImage.shape[0]
		self.currentRawImage = self.currentRawImage[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:]
		self.currentRawImage = cv2.bitwise_and(self.currentRawImage, self.bowl_mask[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:])

		if tracking==True:
			self.track_raw_image()
			self.image_widget.setLabel(self.display_frame)
		else:
			self.image_widget.setLabel(self.currentRawImage)

	@pyqtSlot()
	def toggle_image(self):

		if self.mainWidgetShowsTrack==True:
			self.update_raw_image(False)
			self.mainWidgetShowsTrack=False
		else:
			self.update_raw_image(True)
			self.mainWidgetShowsTrack=True

	@pyqtSlot(int)
	def split_read(self, read_to_split):
		self.highest_index = len(self.active_frame_info.list_of_contour_points)
		self.rts = self.active_frame_info.list_of_contour_points[read_to_split]
		if self.highest_index < self.n_animals:
			new_contour_pt = vc.ContourPoint(self.highest_index+1, self.rts.x,  self.rts.y,  self.rts.c, self.rts.label_config+1)
			self.active_frame_info.add_contour_point(new_contour_pt)
			self.update_raw_image(True)
		else:
			# Error message
			pass

	@pyqtSlot()
	def add_read(self):
		if len(self.active_frame_info.list_of_contour_points) < self.n_animals:
			self.point_adder = vu.PointAdder(self.currentRawImage)
			self.point_adder.define_point_location()
			point = self.point_adder.get_point()
			self.highest_index = len(self.active_frame_info.list_of_contour_points)
			new_contour_pt = vc.ContourPoint(self.highest_index+1, point[0],  point[1], None, 1)
			self.active_frame_info.add_contour_point(new_contour_pt)
			temp = [e for e in self.active_frame_info.list_of_contour_points]
			self.active_frame_info.list_of_contour_points = temp
			for idx, e in enumerate(self.active_frame_info.list_of_contour_points):
				e.id = idx+1
			self.update_raw_image(True)

		else:
			# Error message
			pass

	@pyqtSlot(int)
	def remove_read(self, read_to_remove):
		if len(self.active_frame_info.list_of_contour_points) > 0:
			temp = [e for e in self.active_frame_info.list_of_contour_points if e.id!=read_to_remove]
			self.active_frame_info.list_of_contour_points = temp
			for idx, e in enumerate(self.active_frame_info.list_of_contour_points):
				e.id = idx+1
			self.update_raw_image(True)
		else:
			# Error message
			pass

	def _get_spaced_frames(self, depth):
		self.frameIdxs = np.linspace(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), num=depth)
		self.frameIdxs = [int(e) for e in self.frameIdxs]

	@pyqtSlot()
	def fix_tracking(self):
		self.tracking_fixer.show()
		self.tracking_fixer.move((self.x() + self.width()) / 2, (self.y() + self.height()) / 2)

	def get_cap(self):
		self.cap = cv2.VideoCapture(self.video_address)

	def load_info(self):
		self.thresh, self.large, self.small,self.solidity, self.extent, self.aspect, self.arc,self.bg, self.video_address, self.n_patches, self.n_animals, self.n_frames, self.bowl_mask, self.food_patch_mask, self.mask_centroid = self.experiment_configurator.get_values()
		metadata = {'thresh':self.thresh,'large':self.large, 'small':self.small,
					'solidity':self.solidity,'extent':self.extent, 'aspect':self.aspect, 'arc':self.arc,'bg':self.bg,
					'video_address':self.video_address, 'n_patches':self.n_patches, 'n_animals':self.n_animals,
					'n_frames':self.n_frames, 'bowl_mask':self.bowl_mask, 'food_patch_mask':self.food_patch_mask, 'mask_centroid':self.mask_centroid}
		self.get_cap()
		self.behavior_selector_widget.adjust_size_widgets(self.n_animals)
		self._get_spaced_frames(self.n_frames)
		self.currentFrameIndex = self.frameIdxs[0]

		self.video_information = vc.VideoInformation(self.video_address, metadata)
		for i in range(self.n_frames):
			new_frame = vc.FrameInformation(i+1, self.video_address, self.n_animals, self.frameIdxs[i])
			self.video_information.add_frame(new_frame)
		self.send_frame_status('')

	def new_analysis_cascade(self):
		self.experiment_configurator.show()
		self.experiment_configurator.move((self.x() + self.width()) / 2, (self.y() + self.height()) / 2)

class FixTrackingWindow(QDialog):
	split_read = pyqtSignal(int)
	remove_read = pyqtSignal(int)
	add_read = pyqtSignal()
	def __init__(self,parent=None):
		super(FixTrackingWindow, self).__init__(parent)

		self.pb1 = QPushButton("Split Joined Read")
		self.pb2 = QPushButton("Remove False Read")
		self.pb3 = QPushButton('Add Missed Read')

		self.pb1.setMaximumWidth(140)
		self.pb2.setMaximumWidth(140)
		self.pb3.setMaximumWidth(140)

		self.sp1 = QSpinBox()
		self.sp2 = QSpinBox()
		self.sp1.setMaximumWidth(54)
		self.sp2.setMaximumWidth(54)

		self.donepb = QPushButton('Done')

		self.grid = QGridLayout()
		self.grid.setSpacing(5)
		self.grid.addWidget(self.pb1, 1, 0)
		self.grid.addWidget(self.sp1,1,1)
		self.grid.addWidget(self.pb2, 2, 0)
		self.grid.addWidget(self.sp2, 2,1)
		self.grid.addWidget(self.pb3, 3, 0)
		self.grid.addWidget(self.donepb, 4, 0)
		self.setLayout(self.grid)
		self.resize(300, 300)
		self.setFixedSize(self.size())
		self.setWindowTitle('Fix Tracking')

		self.pb1.clicked.connect(self.split_reads)
		self.pb2.clicked.connect(self.remove_reads)
		self.pb3.clicked.connect(self.add_reads)
		self.donepb.clicked.connect(self.close)

	def split_reads(self):
		if int(self.sp1.value())==0:
			pass
		else:
			self.read_to_split = int(self.sp1.value())-1
			self.split_read.emit(self.read_to_split)
			self.sp1.setValue(0)

	def remove_reads(self):
		if int(self.sp2.value())==0:
			pass
		else:
			self.read_to_remove = int(self.sp2.value())
			self.remove_read.emit(self.read_to_remove)
			self.sp2.setValue(0)


	def add_reads(self):
		self.add_read.emit()

class BehaviorSelectorWidget(QWidget):
	toggleTrack = pyqtSignal()
	newListofBehaviors = pyqtSignal()
	fix_now = pyqtSignal()
	signal_save = pyqtSignal()
	def __init__(self, parent):
		super(BehaviorSelectorWidget, self).__init__(parent)
		self.vbox = QVBoxLayout()
		self.spacer  = QSpacerItem(145,1)
		self.vbox.addSpacerItem(self.spacer)
		self.setLayout(self.vbox)
		self.parent().update_combos.connect(self.update_comboboxes)
		self.mw = self.parent()

	@pyqtSlot()
	def update_comboboxes(self):
		if self.mw.active_frame_info.saved:
			sexes = [e[0] for e in self.mw.active_frame_info.behavior_list]
			behaviors = [e[1] for e in self.mw.active_frame_info.behavior_list]

			for idx, cbb in enumerate(self.list_of_comboboxes):
				index = cbb.findText(behaviors[idx], QtCore.Qt.MatchFixedString)
				if index >= 0:
					cbb.setCurrentIndex(index)
			for idx, scbb in enumerate(self.list_of_scomboboxes):
				index = scbb.findText(sexes[idx], QtCore.Qt.MatchFixedString)
				if index >= 0:
					scbb.setCurrentIndex(index)
		else:
			for idx, cbb in enumerate(self.list_of_comboboxes):
				cbb.setCurrentIndex(0)
				self.list_of_scomboboxes[idx].setCurrentIndex(0)

	def adjust_size_widgets(self, n_individuals):
		self.list_of_comboboxes = []
		self.list_of_scomboboxes = []
		n_individuals = min(n_individuals, 25)
		for i in range(n_individuals):
			label = QLabel()
			label.setText('{}'.format(i+1))

			scombo = QComboBox(self)
			scombo.addItem('-')
			scombo.addItem('F')
			scombo.addItem('M')

			combo = QComboBox(self)
			combo.addItem("Nothing")
			combo.addItem("Courting")
			combo.addItem("Copulation")
			hbox = QHBoxLayout()
			hbox.addWidget(label)
			hbox.addWidget(scombo)
			hbox.addWidget(combo)
			self.list_of_comboboxes.append(combo)
			self.list_of_scomboboxes.append(scombo)
			self.vbox.addLayout(hbox)
		self.saveButton = QPushButton('Save')
		self.toggleButton = QPushButton('Toggle Tracking')
		self.fixButton = QPushButton('Fix Tracking')
		self.fixButton.clicked.connect(self.signal_fix_track)

		self.saved_indicator = QLabel()
		self.saved_indicator.resize(24,24)
		self.saved_indicator.setMaximumWidth(24)
		self.saved_indicator.setStyleSheet('background-color: red')

		spacer_label = QLabel()
		spacer_label.setText('        ')
		hbox = QHBoxLayout()
		hbox.addWidget(spacer_label)
		hbox.addWidget(self.fixButton)
		hbox.addWidget(self.toggleButton)
		self.vbox.addLayout(hbox)
		self.saveButton.clicked.connect(self.log_entries)
		self.toggleButton.clicked.connect(self.toggle_track)
		hbox1 = QHBoxLayout()
		hbox1.addWidget(spacer_label)
		hbox1.addWidget(self.saved_indicator)
		hbox1.addWidget(self.saveButton)
		self.vbox.addLayout(hbox1)

	def signal_fix_track(self):
		self.fix_now.emit()

	def toggle_track(self):
		self.toggleTrack.emit()

	def log_entries(self):
		self.list_of_behaviors = []
		for idx, i in enumerate(self.list_of_comboboxes):
			self.list_of_behaviors.append([str(self.list_of_scomboboxes[idx].currentText()), str(i.currentText())])
		self.newListofBehaviors.emit()
		self.signal_save.emit()
		return None

class ImageWidget(QWidget):
	request_frame_status = pyqtSignal(str)
	def __init__(self, parent):
		super(ImageWidget, self).__init__(parent)
		vbox = QVBoxLayout()
		hbox1 = QHBoxLayout()
		hbox2 = QHBoxLayout()

		self.label = QLabel()
		self.label.move(10, 40)
		self.label.resize(700, 700)
		hbox1.addWidget(self.label)
		self.forwardPB = QPushButton('>>')
		self.backwardPB = QPushButton('<<')
		self.frameLabel = QLabel('                         ')
		self.frameLabel.setMaximumWidth(60)
		hbox2.addWidget(self.backwardPB)
		hbox2.addWidget(self.frameLabel)
		hbox2.addWidget(self.forwardPB)
		vbox.addLayout(hbox1)
		vbox.addLayout(hbox2)
		self.setLayout(vbox)
		testImage = np.zeros((900,900,3),dtype=np.uint8)
		self.setLabel(testImage)
		self.forwardPB.clicked.connect(self.increment_frame_count)
		self.backwardPB.clicked.connect(self.decrement_frame_count)
		self.parent().frameTuple.connect(self.updateFrameLabel)

	def increment_frame_count(self):
		self.request_frame_status.emit('inc')

	def decrement_frame_count(self):
		self.request_frame_status.emit('dec')

	@pyqtSlot(int, int)
	def updateFrameLabel(self, v1, v2):
		self.frameLabel.setText('    {} / {}'.format(v1, v2))

	@pyqtSlot(QImage)
	def setLabel(self, image):

		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		image = image.copy()
		self.qimage = QImage(image, image.shape[0], image.shape[1], QImage.Format_RGB888)
		self.qimage = self.qimage.scaled(900,900)
		pixmap = QPixmap(self.qimage)
		#pixmap = pixmap.scaled(700,700)
		self.imageRef = self.qimage.scaled(900,900)
		self.label.setPixmap(pixmap)
		self.label.show()

def main():
	app = QApplication(sys.argv)
	win = MainWindow()
	win.show()
	app.exec_()

if __name__ == '__main__':
	sys.exit(main())
