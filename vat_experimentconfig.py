"""
Display configuration window and store results provided by user
"""
import os
import cv2
import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import vat_utilities as vu
import vat_ui_warnings as vuw
##
##
##
##
class ExperimentConfigWindow(QtWidgets.QDialog):
	#######################################################################
	# Class to create a QDialog window that allows user to fill in configuraion
	# information for experiment as a whole. This includes:
	# 		1. Selecting a video
	# 		2. Defining chamber and food patch locations
	# 		3. Creating an average background image for use in all frame tracking
	#		4. Defining  thresholds for good tracking
	# 		5. Configuring which user-filled dropdowns will be present

	# Many of these tasks require opening a new thread or ui window
	# In cases other than background subtraction, the function is a member of
	# this class, using  utilites from the vat_utilities module for selection
	# of ellipses, drawing, etc.

	# Because of the higher-compute needed for the background computation,
	# it is necessary to spin this off to a new thread.
	# Doing so was easier by creating a new BGSubtractor class that recieves a
	# 'start' signal from this class, and signals back with the completed matrix,
	# updating a progress bar in the meantime and blocking all other ui tasks
	# until completion. Otherwise, unpredictable ui behavior and crashes would
	# certainly occur.
	#######################################################################

	# Initialize Qt signals to report when the configuration is complete and
	# when background computation should begin. An average background is computed
	# mid-way through experiment configuration in a seperate thread, and therefore needs:
	# 	1. a signal from this class that will be consumed by the BGCalculator class
	#	2. a slot to consume the completed background from the BGCalculator class
	#		when ccomplete
	food_patches_defined = QtCore.pyqtSignal()
	got_values = QtCore.pyqtSignal(int)
	start_bg_comp = QtCore.pyqtSignal(str, int)
	def __init__(self,parent=None):
		# Create buttons, labels and spinboxes for configuration window
		#######################################################################
		super(ExperimentConfigWindow, self).__init__(parent)
		self.pb1 = QtWidgets.QPushButton("Select Video")
		self.pb2 = QtWidgets.QPushButton("Define Chamber")
		self.l3 = QtWidgets.QLabel('N Food Patch')
		self.sp3 = QtWidgets.QSpinBox()
		self.pb4 = QtWidgets.QPushButton("Define Food Patches")
		self.pb5 = QtWidgets.QPushButton("Set Thresholds")
		self.l6 = QtWidgets.QLabel('Chamber D (mm)')
		self.sp6 = QtWidgets.QSpinBox()
		self.sp6.setMinimum(1)
		self.sp6.setMaximum(500)
		self.sp6.setValue(100)
		self.l7 = QtWidgets.QLabel('N Animals')
		self.sp7 = QtWidgets.QSpinBox()
		self.sp7.setMinimum(1)
		self.l8 = QtWidgets.QLabel('N Frames')
		self.sp8 = QtWidgets.QSpinBox()
		self.sp8.setMinimum(1)
		self.sp6.setMaximum(1000)
		self.l9 = QtWidgets.QLabel("N Species")
		self.sp9 = QtWidgets.QSpinBox()
		self.sp9.setMinimum(1)
		self.l10 = QtWidgets.QLabel('Include Sex Info')
		self.cb10 = QtWidgets.QCheckBox()
		self.l11 = QtWidgets.QLabel('Include Courting Partner')
		self.cb11 = QtWidgets.QCheckBox()

		# Make button for saving and sending to main
		self.savepb = QtWidgets.QPushButton('Apply Configuration')

		# Create indicator labels to indicate which fields have been filled
		# and which need attention
		self.indicator_label1 = QtWidgets.QLabel()
		self.indicator_label2 = QtWidgets.QLabel()
		self.indicator_label3 = QtWidgets.QLabel()
		self.indicator_label4 = QtWidgets.QLabel()
		self.indicator_label5 = QtWidgets.QLabel()
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

		# Create a grid layout and add configuration and indicator widgets
		self.grid = QtWidgets.QGridLayout()
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
		self.grid.addWidget(self.l6, 6, 0)
		self.grid.addWidget(self.sp6, 6, 1)
		self.grid.addWidget(self.l7, 7, 0)
		self.grid.addWidget(self.sp7, 7, 1)
		self.grid.addWidget(self.l8, 8, 0)
		self.grid.addWidget(self.sp8,8, 1)
		self.grid.addWidget(self.l9, 9, 0)
		self.grid.addWidget(self.sp9, 9, 1)
		self.grid.addWidget(self.l10, 10,0)
		self.grid.addWidget(self.cb10, 10,1)
		self.grid.addWidget(self.l11, 11,0)
		self.grid.addWidget(self.cb11, 11,1)
		self.grid.addWidget(self.savepb, 12, 1)

		# Set and format the configuration window layout
		self.setLayout(self.grid)
		self.resize(380, 380)
		self.setFixedSize(self.size())
		self.setWindowTitle('Experiment Configuration')
		self.savepb.clicked.connect(self.store_values)

		# Define callbacks for the configuration buttons that depend on
		# other functions
		self.pb1.clicked.connect(self.select_video)
		self.pb2.clicked.connect(self.define_bowl_mask)
		self.pb4.clicked.connect(self.define_food_patch)
		self.pb5.clicked.connect(self.define_thresholds)

		# Create callback to automatically start background computation once
		# arena and foodpatches are defined
		self.food_patches_defined.connect(self.start_background_comp)
##
##
##
	# Specify slot that will recieve calculated background array from BGCalculator class
	@QtCore.pyqtSlot(np.ndarray)
	def on_finished(self, array):
		# Recieve background, destroy progress window which is a member of this
		# class' instance of BGCalculator object, bring the background array in
		# as a member of this class under name 'bg'
		#######################################################################
		self.bg_progress_window.close()
		self.bg = array
##
##
##
	def select_video(self):
		# Use Qt builtin file selector window to allow user to pick video
		#######################################################################
		self.video_address = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Video to Analyze', os.getcwd())[0]

		# Set indicator label to green when done
		self.indicator_label1.setStyleSheet('background-color: green')
##
##
##
	def define_bowl_mask(self):
		# Configuration step to create a mask that should only show valid bowl area
		#######################################################################
		try:
			# Create an instance of vat_utilities.EllipseDrawer
			ed = vu.EllipseDrawer(self.video_address)

			# Use the instances member function to make new ui window and select
			# and return bowl parameters
			ed.define_bowl_mask()

			# Call the instances getter function to return  the bowl mask and centroid of the bowl
			self.bowl_mask, self.mask_centroid = ed.get_bowl_mask()

			# Set indicator label to green when done
			self.indicator_label2.setStyleSheet('background-color: green')
##
##
##
		except AttributeError:
			# Invoke error window to prevent crash if mask generation is attempted without a valid video
			#######################################################################
			msg = 'Make sure a valid video is selected'
			self.error = vuw.ErrorMsg(msg)
			self.error.show()
##
##
##
	@QtCore.pyqtSlot()
	def start_background_comp(self):
		# Configuration step to compute average background for future processing
		# on all analysis frames. Does not compute background in this function,
		# but starts and performs necessary signalling with BGCalculator object
		# as well as manages the progress bar and blocking
		#######################################################################
		# Create a new progress bar dialog window to update progress on background computation
		self.bg_progress_window = QtWidgets.QProgressDialog(self)

		# Immediately hide that window until it is ready
		self.bg_progress_window.hide()
		self.bg_progress_window.setLabelText('Computing Average Background Features')

		# Spin off new process thread and start it without assignment
		thread = QtCore.QThread(self)
		thread.start()

		# Create an instance of BGCalculator class
		self.bgc = BGCalculator()

		# Add a callback from this class' start_bg_comp signal to the BGCalculator instance's
		# calculateBGwithProgress function, which calculates the background while updating this class'
		# bg_progress_window
		self.start_bg_comp.connect(self.bgc.calculateBGwithProgress)
		self.bgc.progressChanged.connect(self.bg_progress_window.setValue)

		# Add a callback so that when BGCalculator instance signals they are finished,
		# progress window is destroyed and background matrix is read in
		self.bgc.finished.connect(self.on_finished)

		# Add a callback so that when computation starts, suppressed progress window is shown
		self.bgc.started.connect(self.bg_progress_window.show)

		# Move BGCalculator instances load to the empty thread
		self.bgc.moveToThread(thread)

		# Send the signal to start background computation
		self.start_bg_comp.emit(self.video_address, self.mask_centroid[0])
##
##
##
	def define_food_patch(self, patches):
		# Configuration step to create a logical mask that should only include
		# the food patches. Used to generate metrics
		#######################################################################
		try:
			# Create an instance of vat_utilities.EllipseDrawer
			ed = vu.EllipseDrawer(self.video_address)

			# Read from spinbox how many patches need to be defined
			self.n_food_patch = int(self.sp3.value())

			# If no patches need to be defined, food_patch_mask is NoneType
			if self.n_food_patch==0:
				self.food_patch_mask=None
				self.food_patch_centroids = None

			else:
				# If > 0 patches need to be defined, invoke the vat_utilities.EllipseDrawer instances
				# foodpatch mask function
				ed.define_food_patches(self.n_food_patch)
				self.food_patch_mask, self.food_patch_centroids = ed.get_food_patches()

				height = self.food_patch_mask.shape[0]
				self.food_patch_mask = self.food_patch_mask[:, int(self.mask_centroid[0]-(height/2)):int(self.mask_centroid[0]+(height/2)),:]


			# Set indicator label to green
			self.indicator_label3.setStyleSheet('background-color: green')

			# Signal that the last piece of information needed to compute
			# background is complete and therefore it can automatically start
			self.food_patches_defined.emit()

		except AttributeError:
			# Invoke error window to prevent crash if mask generation is attempted without a valid video
			msg = 'Make sure a valid video is selected and chamber is defined'
			self.error = vuw.ErrorMsg(msg)
			self.error.show()
##
##
##
	def define_thresholds(self):
		# Configuration step to let user check contour filtering settings
		# and adjust to specific lighting conditions using UI with sliders
		#######################################################################
		try:
			# Create an instance of vat_utilities.Thresholder
			thresholder = vu.Thresholder(self.video_address, self.bg, self.mask_centroid[0], self.bowl_mask)

			# When the process is done and window is closed, save all information returned from
			# instances 'get_values' function
			self.thresh, self.small, self.large, self.solidity, self.extent, self.aspect, self.arc = thresholder.get_values()

			# Set indicator label to green
			self.indicator_label4.setStyleSheet('background-color: green')

		except AttributeError:
			# Invoke error window to prevent crash if thresholding is attempted without setting chamber and food patch
			msg = 'Chamber and food patches must be defined before setting thresholds'
			self.error = vuw.ErrorMsg(msg)
			self.error.show()
##
##
##
	def store_values(self):
		# When the user has added all necessary information to configure a trial,
		# and all necessary subprocesses have completed, all information is saved
		# as attributes of this class that will be available to other parts of the program
		# through this class' 'get_values' function
		#######################################################################
		# Store spinbox and checkbox values
		self.n_patches = int(self.sp3.value())
		self.chamber_d = int(self.sp6.value())
		self.n_animals = int(self.sp7.value())
		self.n_frames = int(self.sp8.value())
		self.n_species = int(self.sp9.value())
		self.include_sex = self.cb10.isChecked()
		self.include_courting_partner = self.cb11.isChecked()

		try:
			# Store threshold parameters and close wwindow
			self.thresh, self.large, self.small,self.solidity, self.extent, self.aspect, self.arc, self.bg, self.video_address, self.n_patches, self.n_animals, self.n_frames, self.bowl_mask, self.food_patch_mask, self.mask_centroid, self.include_sex, self.n_species, self.chamber_d, self.include_courting_partner, self.food_patch_centroids

			# Close this configuration window in ui without destroying the object
			# Information can still be accessed even through dialog window is not visible
			self.close()

			# Signal to MainWindow that all cinfiguration values have been defined
			self.got_values.emit(1)

		except AttributeError:
			# Throw an error if there is missing configration information
			msg = 'Make sure all parameters are defined even if they are not important for present trial'
			self.error = vuw.ErrorMsg(msg)
			self.error.show()
##
##
##
	def get_values(self):
		# Roll return values into a dictionary and return
		#######################################################################
		vals = {'thresh'					:self.thresh,
				'large'						:self.large,
				'small'						:self.small,
				'solidity'					:self.solidity,
				'extent'					:self.extent,
				'aspect'					:self.aspect,
				'arc'						:self.arc,
				'bg'						:self.bg,
				'video_address'				:self.video_address,
				'n_patches'					:self.n_patches,
				'n_animals'					:self.n_animals,
				'n_frames'					:self.n_frames,
				'bowl_mask'					:self.bowl_mask,
				'food_patch_mask'			:self.food_patch_mask,
				'food_patch_centroids'		:self.food_patch_centroids,
				'mask_centroid'				:self.mask_centroid,
				'include_sex'				:self.include_sex,
				'n_species'					:self.n_species,
				'chamber_d'					:self.chamber_d,
				'include_courting_partner'	:self.include_courting_partner
				}
		return vals
##
##
##
##
class BGCalculator(QtCore.QObject):
	#######################################################################
	# Inherited class from QtCore.QObject, used to perform background computation
	# and signalling of progress during the experiment configuration stage.
	# Only signals and consumes from ExperimentConfigWindow and is not used
	# after experiment configuration
	#######################################################################
	# Define signals that will update ExperimentConfigWindow's progress bar
	progressChanged = QtCore.pyqtSignal(int)
	started = QtCore.pyqtSignal()

	# Define the signal to send finished background array to the ExperimentConfigWindow
	finished = QtCore.pyqtSignal(np.ndarray)

	def calculateAndUpdate(self, done, total):
		# Calculate the percent of the computation that has been completed and
		# Signal to the ExperimentConfigWindow
		#######################################################################
		# get a 0-100 integer representing the proportion of frames already computed into background
		progress = int(round((done / float(total)) * 100))

		# Send signal that carries progress value to ExperimentConfigWindow object's progressbar
		self.progressChanged.emit(progress)
##
##
##
	@QtCore.pyqtSlot(str, int)
	def calculateBGwithProgress(self, video_address, center, start=0):
		# Compute average background of a sampling of frames and send progress
		#######################################################################
		# Grab an arbitrary early frame towards the beginning of the video for purpose of frame geometry
		cap = cv2.VideoCapture(video_address)
		_, frame =cap.read()
		cap.set(cv2.CAP_PROP_POS_FRAMES, 80)

		# Get frame height and use it to grab left and right boungs for chamber
		# FIXME: This is sloppy, and probably the reason some frames are cut off on the bottom
		# 			during analysis. Not a problem if dish with warping is taller than wide
		height = frame.shape[0]
		xl, xr = center-int(height/2), center+int(height/2)

		# This is the number of samples that will be taken across the entire video
		# to construct a good background image.
		# For 1 hour videos, 80 was found through trial and error to produce a good background frame without
		# excessive compute time. Could probably experiment with using fewer
		depth = 80
		blank = []

		# Get a list of evenly spaced frame indexes over the framecount of the video
		frameIdxs = np.linspace(start, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-100), num=depth)
		frameIdxs = [int(e) for e in frameIdxs]

		# Exclude the last frame as the cv2.CAP_PROP_FRAME_COUNT provided framecount is an inexact measuure,
		# and grabbing the last one frequently results in calling an out-of-index frame and crashing
		frameIdxs = frameIdxs[:-2]

		# Tell the ExperimentConfigWindow object that background subtraction has begun
		self.started.emit()

		for idx,i in enumerate(frameIdxs):
			# Iterate through the list of spaced indices and grab the source frame at that index
			cap = cv2.VideoCapture(video_address)
			cap.set(cv2.CAP_PROP_POS_FRAMES, i)
			_, frame = cap.read()

			# Chop it to the window size
			frame = frame[:,xl:xr,:]

			# Convert it to mono
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Send signal to update progress bar
			self.calculateAndUpdate(idx,len(frameIdxs))

			# Append the frame to the temporary list that will be used to take an average
			blank.append(frame[:,:])

		# Convert list to array
		blank = np.asarray(blank)

		# Take mean across sample axis
		bg = np.mean(blank, axis=0)

		# !! When working with photos, make sure all mats are in 8-bit unsigned integers!!
		bg = bg.astype(np.uint8)

		# Signal that computation is done and carry it to the ExperimentConfigWindow
		self.finished.emit(bg)
		bgDone = True
