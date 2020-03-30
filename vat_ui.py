"""
UI Main window functionality
"""
import sys
import os
import cv2
import pickle
import threading
import numpy as np
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets

import vat_core as vc
import vat_widgets as vw
import vat_utilities as vu
import vat_fixtracking as vft
import vat_ui_warnings as vuw
import vat_experimentconfig as vec
##
##
##
##
class MainWindow(QtWidgets.QMainWindow):
	#######################################################################
	# Main UI class with both app and ui functionality.
	# Serves as a centerpoint and parent to other widgets
	#######################################################################
	# Define signals
	frameTuple = QtCore.pyqtSignal(int, int)	# An integer 2-tuple that contains current frame index and total frame count
	update_combos = QtCore.pyqtSignal()			# Command to update and re-render comboboxes
	log_now = QtCore.pyqtSignal()				# Commnad to take current state of comboboxes in BehaviorSelectorWidget object
												# and save them as attributes
	def __init__(self, parent=None):
		# Does essential startup by instantiating necessary widget classes,
		# setting ui elements and callbacks
		#######################################################################
		super(MainWindow, self).__init__(parent)

		# Create instances of necessary widget classes
		self._initialize_widgets()

		# Define mainwindow layout
		_widget = QtWidgets.QWidget()
		_layout = QtWidgets.QHBoxLayout(_widget)

		# Only imagewidget and behavior_selector_widget are in layout, other
		# widgets mentioned above are conditionally-displayed dialogs
		_layout.addWidget(self.image_widget)
		_layout.addWidget(self.behavior_selector_widget)
		self.setCentralWidget(_widget)

		# Add file menu with necessary actions such as save and new
		self._add_filemenu()

		# Set callbacks for the main widgets (image and behavior selector)
		self._set_widget_callbacks()

		# Initialize indices and other startup variables
		self.currentFrame = 1
		self.n_frames = 0
		self.frameLabel = '{} / {}'.format(self.currentFrame+1, self.n_frames)
		self.mainWidgetShowsTrack = True
		self.colorMap = [(153,255,153),(204,255,153),(255,153,255),(204,153,255),(51,51,255),(51,153,255),(51,255,255),(51,255,153),(255,51,255),(153,51,255),(153,153,255),(153,204,255),(255,255,153),(255,204,153),(255,153,204),(51,255,51),(153,255,51),(255,255,51),(255,153,51),(255,51,51),(255,51,153),(153,255,255),(153,255,204),(0,0,204),(0,204,102),(204,204,0),(204,102,0),(153,153,0),(0,204,0),(153,0,153)]
		self.setFixedSize(1250,1000)
		self.warned = False
		self.wrote_patch_mask = False
		self.attempt = 0
##
##
##
	def _initialize_widgets(self):
		# Makes instances of necessary widgets and saves them as attributes
		#######################################################################
		# Initial configuration manager
		self.experiment_configurator = vec.ExperimentConfigWindow()

		# Dialog to manager user fixing of automatic tracking reads
		self.tracking_fixer = vft.FixTrackingWindow(self)

		# Manges dropdowns for behavior, sex, courting partner, species
		self.behavior_selector_widget = vw.BehaviorSelectorWidget(self)

		# Manages display&update to main ui image
		self.image_widget = vw.ImageWidget(self)
##
##
##
	def _add_filemenu(self):
		# Creates filemenu with save/new actions and adds to MainWindow
		#######################################################################
		# Define actions across top action bar
		newAct = QtWidgets.QAction('New Analysis', self)

		# Set callback for New Analysis option
		newAct.triggered.connect(self.new_analysis_cascade)
		self.experiment_configurator.got_values.connect(self.load_info)
		menubar = self.menuBar()

		# Add New analysis option to File menu
		fileMenu = menubar.addMenu('File')
		fileMenu.addAction(newAct)

		# Add save analysis to file menu, set its callback
		saveAct = QtWidgets.QAction('Save Analysis', self)
		saveAct.triggered.connect(self.save_analysis)
		fileMenu.addAction(saveAct)

		# Add save analysis to file menu, set its callback
		loadAct = QtWidgets.QAction('Load from Last Break', self)
		loadAct.triggered.connect(self.load_cache)
		fileMenu.addAction(loadAct)
##
##
##
	def _set_widget_callbacks(self):
		# Set callbacks for dropdown menus widget
		#######################################################################
		# Callback to show/hide tracking overlays
		self.behavior_selector_widget.toggleTrack.connect(self.toggle_image)

		# Callback to take data from dropdowns and save it as attribute to storage class' behavior_list objects
		self.behavior_selector_widget.newListofBehaviors.connect(self.update_list_of_behaviors)

		# Callback to display fix tracking dialog
		self.behavior_selector_widget.fix_now.connect(self.fix_tracking)

		# Callback to save current data to pickle cache
		self.behavior_selector_widget.signal_save.connect(self.save_to_cache)

		# Callback to emit from log_now signal, which starts the log_entries routine
		self.image_widget.signal_save.connect(self.send_log_now)

		# Callback to connect the imagewidgets request for index (current frame/n_frames)
		# with this widgets function to send index
		self.image_widget.request_frame_status.connect(self.send_frame_status)
##
##
##

	def check_and_warn(self):
		# Check to ensure if all frames and all animals within each frame have been accounted for
		# self.warned 'latches' and will only show warning once as it is a non-critical error
		#######################################################################
		if self.warned==False:
			frame_warn = False
			id_warn = False

			# Initialize warning as an empty list and add strings based on the
			# categories of present errors
			warning = []

			# Iterate through FrameInformation objects
			for frame in self.video_information.get_frame_list():

				# If warning already happened, pass
				if id_warn == True:
					pass

				else:
					# If in the current frame, not all animals have been accounted
					# for, add this message
					if len(frame.list_of_contour_points)<self.n_animals:
						id_warn = True
						warning.append('There are frames where the number of tracked animals is less than {}'.format(self.n_animals))

				# If not all fraames have been analyzed, add this message
				if frame.list_of_contour_points==[] and frame_warn == False:
					frame_warn = True
					warning.append('There are frames that were not visited')
			msg=''

			# Append warning messages and construct a single string to display
			for warn in warning:
				msg = msg + warn + '\n\n'
			msg = msg + 'You may still save, but the unvisited frames will not be represented in the output CSV'

			# Create / show an instance of the Warnin object
			if frame_warn or id_warn:
				self.warning = vuw.WarningMsg(msg)
				self.warning.show()
			self.warned = True
		self.attempt += 1
##
##
##
	def save_analysis(self):
		# End-of-analysis save routine
		# Write data to pickle and create readable output csv and graphics
		#######################################################################
		try:
			self.name

			# Make sure all frames / animals accounted for
			self.check_and_warn()

			# Give one warning and then save away
			if self.warned and self.attempt==1:
				pass

			else:
				# If the last/current frame hasn't been cached, send signal to do so
				if self.active_frame_info.saved==False:
					self.send_log_now()

				# Dump the VideoInformation storage container to a pickle
				with open(os.path.join(os.getcwd(), 'data','{}/{}.pkl'.format(self.name,self.name)), 'wb') as f:
					pickle.dump(self.video_information, f)
				picklename = os.path.join(os.getcwd(), 'data','{}/{}.pkl'.format(self.name,self.name))

				# Create an instance of the DataWriter object
				dw = vu.DataWriter(picklename)

				# Generate informative rows using the DataWriter make_rows method
				dw.make_rows()

				# Write rows to output format using the DataWriter write_csv method
				dw.write_csv()


		except AttributeError:
			# Throw error if a configured experiment does not exist
			msg = 'You must configure a new experiment in order to save'
			self.error = vuw.ErrorMsg(msg)
			self.error.show()
##
##
##
	@QtCore.pyqtSlot()
	def send_log_now(self):
		# Utility to emit the log_now signal, which invokes log_entries routine
		#######################################################################
		self.log_now.emit()
##
##
##
	def make_data_folder(self):
		# Startup utility function to check if data and experiment_cache folders
		# exist, and make them if not
		#######################################################################
		# Make data folder if it doesnt exist
		if 'data' not in os.listdir(os.getcwd()):
			os.mkdir(os.path.join(os.getcwd(), 'data'))

		# Make experiment_cache folder if it doesnt exist
		if 'experiment_cache' not in os.listdir(os.getcwd()):
			os.mkdir(os.path.join(os.getcwd(), 'experiment_cache'))

		# Make a name attribute that is the video name with no file extensions
		self.name = (self.video_address.split('/')[-1]).split('.')[0]

		# If that names is not in the data folder, make a folder for it to store stuff
		if self.name not in os.listdir(os.path.join(os.getcwd(), 'data')):
			os.mkdir(os.path.join(os.getcwd(), 'data', self.name))

		# If there is no labelled_photos folder, make that
		if 'labelled_photos' not in os.listdir(os.path.join(os.getcwd(), 'data', self.name)):
			os.mkdir(os.path.join(os.getcwd(), 'data', self.name,'labelled_photos'))
##
##
##
	@QtCore.pyqtSlot()
	def save_to_cache(self):
		# Sets the metadata's last-analyzed-frame to the current frame,
		# and saves all information to a pickle. Called every time a frame is advanced
		# in the analysis
		#######################################################################
		# We do this before saving so the metadata dictionary will reflect progress
		self.video_information.metadata['last_frame'] = self.currentFrame

		# Dump the VideoInformation storage object into a pickle
		with open(os.path.join(os.getcwd(), 'experiment_cache','cache.pkl'), 'wb') as f:
			pickle.dump(self.video_information, f)
##
##
##
	def load_cache(self):
		# Opens the pickle in the experiment_cache, and populates attributes with
		# the contents to the metadata dictionary. USed to load a past Analysis
		# or recover progress from a crash
		#######################################################################
		try:
			with open(os.path.join(os.getcwd(), 'experiment_cache','cache.pkl'), 'rb') as f:
				self.video_information = pickle.load(f)
				self.currentFrame = 1
				self.thresh = self.video_information.metadata['thresh']
				self.bg = self.video_information.metadata['bg']
				self.large = self.video_information.metadata['large']
				self.small = self.video_information.metadata['small']
				self.solidity = self.video_information.metadata['solidity']
				self.extent = self.video_information.metadata['extent']
				self.aspect = self.video_information.metadata['aspect']
				self.arc = self.video_information.metadata['arc']
				self.video_address = self.video_information.metadata['video_address']
				self.n_patches = self.video_information.metadata['n_patches']
				self.n_animals = self.video_information.metadata['n_animals']
				self.n_frames = self.video_information.metadata['n_frames']
				self.bowl_mask = self.video_information.metadata['bowl_mask']
				self.food_patch_mask = self.video_information.metadata['food_patch_mask']
				self.food_patch_centroids = self.video_information.metadata['food_patch_centroids']
				self.mask_centroid = self.video_information.metadata['mask_centroid']
				self.include_sex = self.video_information.metadata['include_sex']
				self.n_species = self.video_information.metadata['n_species']
				self.chamber_d = self.video_information.metadata['chamber_d']
				self.include_courting_partner = self.video_information.metadata['include_courting_partner']
				self.currentFrame = self.video_information.metadata['last_frame']
				self.warned = False
				self.attempt = 0

			# Make data folder and Initialize video capture object
			self.make_data_folder()
			self.get_cap()

			# Set which dropdowns will be included in the behavior_selector_widget
			self.behavior_selector_widget.set_include_sex_and_species(self.include_sex, self.n_species, self.include_courting_partner)
			self.behavior_selector_widget.adjust_size_widgets(self.n_animals)

			# Initializes frame indices, count (0/N), and shows the current frame in ImageWidget matrix
			self._get_spaced_frames(self.n_frames)
			self.send_frame_status('')
			self.update_raw_image(True)

		except (KeyError, FileNotFoundError) as err:
			msg = 'Could not properly load a cached experiment. Start a new analysis and try again.'
			self.warning = vuw.WarningMsg(msg)
			self.warning.show()

##
##
##
	@QtCore.pyqtSlot(str)
	def send_frame_status(self, id):
		# Increment or decrement the current frame index and update
		# frame progress label.
		#######################################################################
		# If action id = 'inc', increment the frame count unless it exceeds
		# the number of frames intended to be analyzed. Invoked with >> button
		if id=='inc':
			self.currentFrame += 1
			if self.currentFrame > self.n_frames:
				self.currentFrame = self.n_frames

		# If action id = 'dec', decrement the frame count unless it is
		# less than 1. Invoked with << button
		elif id == 'dec':
			self.currentFrame -= 1
			if self.currentFrame < 1:
				self.currentFrame= 1
		else:
			pass

		# After making necessary inc/dec changes, update the current index attribute
		self.currentFrameIndex = self.frameIdxs[self.currentFrame-1]
		if self.currentFrame==self.n_frames:
			self.currentFrameIndex = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)-300)

		# Send to the ImageWidget a 2-tuple of current frame and nummber of Frames
		# as a means of progress through the analaysis
		self.frameTuple.emit(self.currentFrame, self.n_frames)

		# Set the videocapture object to the video frame referenced by the current index
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrameIndex)

		# Make an attribute for mainwindow to remember that the ImageWidget will be showing tracking
		self.mainWidgetShowsTrack=True

		# Re-render frame after making changes to index
		self.update_raw_image(True)

		# Emit command to update comboboxes for new frame
		self.update_combos.emit()
##
##
##
	def track_raw_image(self):
		# Performs automatic tracking on the current frame and draws necessary overlays
		#######################################################################
		# Define local utility function
		def _get(cnt):
			# Takes cv2.contour type "cnt" as input and computes various statistics
			# to be used for filtering, based on filtering thresholds set during
			# initial configuration
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

		# If the frames has not been tracked yet, perform the automatic contour detection/filter routine
		if self.active_frame_info.tracked==False:
			# Copy and convert to black and white
			self.tracked_image = self.currentRawImage.copy()
			self.tracked_image = cv2.cvtColor(self.tracked_image, cv2.COLOR_BGR2GRAY)

			# Subtract background and increase all channels slightly
			self.tracked_image=self.tracked_image-self.bg + 10
			height = self.currentRawImage.shape[0]

			# Grab appropriately-sized chunk (matching size of analysis frame) of the bowl mask matrix
			self.bm = cv2.cvtColor(self.bowl_mask[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:], cv2.COLOR_BGR2GRAY)

			# Apply logical masking on image using the background mask bm
			self.tracked_image = cv2.bitwise_and(self.tracked_image, self.bm)

			# Perform binary threshold on masked/modified image to get black/white
			_,self.threshed_frame = cv2.threshold(self.tracked_image,self.thresh,255,cv2.THRESH_BINARY)

			# Invert
			self.imagem = cv2.bitwise_not(self.threshed_frame)

			# Perform cv2 contour detection algo
			self.contours, hier = cv2.findContours(self.imagem, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			# Filter contours based on attributes, save valid one to valids list attribute
			self.valids = [cnt for cnt in self.contours if (_get(cnt)[3] > self.small) and (_get(cnt)[3] < self.large) and _get(cnt)[0] < self.aspect and _get(cnt)[1] < self.extent and _get(cnt)[2] > self.solidity and _get(cnt)[-1]<self.arc]

			# Shallow copy current raw image as display frame, for drawing
			# Copy is necessary so drawing actions are not persistent with the raw image frame
			self.display_frame = self.currentRawImage

			# For the list of valid contours, get x/y positions of centroids using mass moment of intertia,
			# assuming pixels have small identical masses
			for idx, c in enumerate(self.valids):
				M = cv2.moments(c)
				if M['m00'] != 0:
					cx = np.float16(M['m10']/M['m00'])
					cy = np.float16(M['m01']/M['m00'])

					# Save x/y coordinate info into ContourPoint storage class
					contour_pt = vc.ContourPoint(idx+1, cx, cy, c, 1)

					# Save the newly created ContourPoint object to the current FrameInformation storage object
					self.active_frame_info.add_contour_point(contour_pt)

					# Draw an outline aroung each valid animal contour in a different color on the display_frame mat
					self.display_frame = cv2.drawContours(self.display_frame, [self.valids[idx]], -1, self.colorMap[idx], 1)

					# Draw the id number of each animal contour in a different color on the display_frame mat
					self.display_frame = cv2.putText(self.display_frame,"{}".format(contour_pt.id),(int(cx+10),int(cy+10)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)
			self.active_frame_info.tracked = True

		else:
			# If frame has already been tracked, load the list of ContourPoints associated
			# with the current FrameInformation object.
			self.display_frame = self.currentRawImage
			for idx, c in enumerate(self.active_frame_info.get_list_of_contour_points()):

				# See if any labels need spatial reconfiguration to not overlap
				if c.label_config==1:
					xp, yp = 10,10
				elif c.label_config==2:
					xp, yp = -10,-10
				elif c.label_config==3:
					xp, yp = 10,-10
				elif c.label_config==4:
					xp, yp = -10,10

				if c.c is not None:
					# if the contour is not a nonetype (meaning it was tracked automatically),
					# draww a perimeter around each valid animal contour and add add a text ID label
					self.display_frame = cv2.drawContours(self.display_frame, [c.c], -1, self.colorMap[idx], 1)
					self.display_frame = cv2.putText(self.display_frame,"{}".format(c.id),(int(c.x+xp),int(c.y+yp)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)

				else:
					# if the contour IS a nonetype, meaning it was added manually at some point
					# with the fix_tracking functionality, there is no perimeter to draw so a point is
					# added at the x/y coordinate and a text id label is added as well
					self.display_frame = cv2.circle(self.display_frame,(c.x,c.y),4,self.colorMap[idx],1)
					self.display_frame = cv2.putText(self.display_frame,"{}".format(c.id),(int(c.x+xp),int(c.y+yp)), cv2.FONT_HERSHEY_PLAIN, 2, self.colorMap[idx], thickness = 2)

		# Write the labelled photo to the labelled_photos folder each time it is modified or a new one is created
		cv2.imwrite(os.path.join(os.getcwd(), 'data', self.name,'labelled_photos','{}.jpg'.format(self.active_frame_info.index)),self.display_frame)

		if self.food_patch_mask is not None and not self.wrote_patch_mask:
			cv2.imwrite(os.path.join(os.getcwd(), 'data', self.name,'labelled_photos','{}_food_mask.jpg'.format(self.active_frame_info.index)),self.display_frame*self.food_patch_mask)
			self.wrote_patch_mask = True
##
##
##
	@QtCore.pyqtSlot()
	def update_list_of_behaviors(self):
		# Bring unsaved status of dropdowns to be saved in persistent storage
		# containers. Both tracked positions and qualitative label assessments
		#######################################################################
		# Indicate that the dropdowns for this frame have been saved
		self.active_frame_info.saved = True
		self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: green')

		# Copy the list of dropdown statuses and x/y's from the behavior_selector_widget to the
		# behavior list attribute of the current FrameInformation object
		self.active_frame_info.behavior_list = self.behavior_selector_widget.list_of_behaviors
		self.active_frame_info.position_list = [[e.id, e.x, e.y] for e in self.active_frame_info.list_of_contour_points]
##
##
##
	def update_raw_image(self, tracking=True):
		# Save the current state of things and update the image in the uis
		# main label with tracking overlays (or not)
		#######################################################################
		# Get the current FrameInformation object
		self.active_frame_info = self.video_information.get_frame_list()[self.currentFrame-1]

		# Display the indicator label based on saved status
		if self.active_frame_info.saved:
			self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: green')
		else:
			self.behavior_selector_widget.saved_indicator.setStyleSheet('background-color: red')

		# Set the VideoCapture object to the correct frame index and read a frame
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrameIndex)
		_, self.currentRawImage = self.cap.read()
		height = self.currentRawImage.shape[0]

		# Crop out the desired area and mask
		self.currentRawImage = self.currentRawImage[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:]
		self.currentRawImage = cv2.bitwise_and(self.currentRawImage, self.bowl_mask[:, (self.mask_centroid[0]-int(height/2)):(self.mask_centroid[0]+int(height/2)),:])

		# Call ImageWidget object's function to update the label,
		# with or without tracking based on desired condition
		if tracking==True:
			self.track_raw_image()
			self.image_widget.setLabel(self.display_frame)
		else:
			self.image_widget.setLabel(self.currentRawImage)
##
##
##
	@QtCore.pyqtSlot()
	def toggle_image(self):
		# Flips the state of whether or not the main ImageWidget is showing tracking overlays
		#######################################################################
		cv2.destroyAllWindows()
		if self.mainWidgetShowsTrack==True:

			# Saves entries before modifying, was resetting to defaults without this
			self.log_now.emit()

			# Call to update ImageWidget without overlays
			self.update_raw_image(False)
			self.mainWidgetShowsTrack=False

		else:
			# Call to update ImageWidget with overlays
			self.update_raw_image(True)
			self.mainWidgetShowsTrack=True
##
##
##
	def test_last_frame(self):
		# Hacky check. When grabbing a frame by index using the cv2.CAP_PROP_POS_FRAMES
		# attribute, the last frame is often erroneous. Want to know what the last frame
		# corresponding to a valid matrix is.
		#######################################################################
		# Until proven Otherwise, assume cv2.CAP_PROP_POS_FRAMES gave a good result
		done = False
		lastFrame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

		while not done:
			try:
				# Check if settng capture to last frame and reading throws exception
				self.cap.set(cv2.CAP_PROP_POS_FRAMES,lastFrame)
				_, r = self.cap.read()
				r.shape
				done=True
				return lastFrame

			except AttributeError:
				# If exception thrown, move back by 1 second. Repeat until valid
				# frame is grabbed
				lastFrame = lastFrame - 30
##
##
##
	def _get_spaced_frames(self, depth):
		# Get a list of length: depth spaced indices for grabbing frames in analysis
		#######################################################################
		# Perform check to get last valid frame
		lastFrame = self.test_last_frame()

		# Calcluate a depth-dimensional evenly spaced array, and save as a list
		self.frameIdxs = np.linspace(0, lastFrame, num=depth)
		self.frameIdxs = [int(e) for e in self.frameIdxs]
##
##
##
	@QtCore.pyqtSlot()
	def fix_tracking(self):
		# Display the fix_tracking dialog window
		#######################################################################
		self.tracking_fixer.show()
		self.tracking_fixer.move((self.x() + self.width()) / 2, (self.y() + self.height()) / 2)
##
##
##
	def get_cap(self):
		# Create a video capture object with the analysis address
		#######################################################################
		self.cap = cv2.VideoCapture(self.video_address)
##
##
##
	def load_info(self):
		# Load information from ExperimentConfigWindow
		#######################################################################
		# Used ExperimentConfigWindow's get_data() to return all information set there
		vals = self.experiment_configurator.get_values()
		self.thresh = vals['thresh']
		self.large = vals['large']
		self.small = vals['small']
		self.solidity = vals['solidity']
		self.extent = vals['extent']
		self.aspect = vals['aspect']
		self.arc = vals['arc']
		self.bg = vals['bg']
		self.video_address = vals['video_address']
		self.n_patches = vals['n_patches']
		self.n_animals = vals['n_animals']
		self.n_frames = vals['n_frames']
		self.bowl_mask = vals['bowl_mask']
		self.food_patch_mask = vals['food_patch_mask']
		self.food_patch_centroids = vals['food_patch_centroids']
		self.mask_centroid = vals['mask_centroid']
		self.include_sex = vals['include_sex']
		self.n_species = vals['n_species']
		self.chamber_d = vals['chamber_d']
		self.include_courting_partner = vals['include_courting_partner']

		# Store information in the metadata dictionary
		metadata = vals

		# Check to make sure data folders exist
		self.make_data_folder()

		# Reinitialize the cv2 VideoCapture object
		self.get_cap()

		# Based on user entries, render the requested dropdown boxes
		self.behavior_selector_widget.set_include_sex_and_species(self.include_sex, self.n_species, self.include_courting_partner)
		self.behavior_selector_widget.adjust_size_widgets(self.n_animals)

		# Get a list of spaced frames for analysis, based on user-desired number
		self._get_spaced_frames(self.n_frames)

		# Initialize frame index counter, create main data storage class VideoInformation
		self.currentFrameIndex = self.frameIdxs[0]
		self.video_information = vc.VideoInformation(self.video_address, metadata)
		self.video_information.metadata['frameIdxs'] = self.frameIdxs

		# Iterate through the number of desired frames, make a blank FrameInformation
		# data storage object, add it to the VideoInformation object
		for i in range(self.n_frames):
			new_frame = vc.FrameInformation(i+1, self.video_address, self.n_animals, self.frameIdxs[i])
			self.video_information.add_frame(new_frame)

		# Keep frame progress indicator blank for now
		self.send_frame_status('')
##
##
##
	def new_analysis_cascade(self):
		# Display the ExperimentConfigWindow dialog to let user configure
		#######################################################################
		self.experiment_configurator.show()
		self.experiment_configurator.move((self.x() + self.width()) / 2, (self.y() + self.height()) / 2)
##
##
##
def main():
	# Run ui on main thread
	#######################################################################
	app = QtWidgets.QApplication(sys.argv)
	win = MainWindow()
	win.show()
	app.exec_()

if __name__ == '__main__':
	sys.exit(main())
