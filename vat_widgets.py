"""
2 main widgets on the mainwindow: dropdowns for annotation and main
image display with tracking overlays
"""
import cv2
import numpy as np
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
##
##
##
##
class BehaviorSelectorWidget(QtWidgets.QWidget):
	#######################################################################
	# One of the two main widgets on the main window, this is the right-side
	# array of dropdown menus allowing user to manually annotate behavior
	# Functions update ui rendering and storage/sending of annotated data
	# to storage classes
	#######################################################################
	# Define signals
	toggleTrack = QtCore.pyqtSignal()			# Toggle tracked/untracked main image
	newListofBehaviors = QtCore.pyqtSignal()	# Behavior dropdown change detected
	fix_now = QtCore.pyqtSignal()				# Start tracking-fixing window
	signal_save = QtCore.pyqtSignal()			# Signal to save to pickle
	def __init__(self, parent):
		# Initilaize ui components and data structures and add to layout
		#######################################################################
		super(BehaviorSelectorWidget, self).__init__(parent)

		# UI components
		self.vbox = QtWidgets.QVBoxLayout()
		self.spacer  = QtWidgets.QSpacerItem(145,1)
		self.vbox.addSpacerItem(self.spacer)
		self.setLayout(self.vbox)

		# Make mw attribute (mainwindow) reference to parent MainWindow object
		self.mw = self.parent()

		# Set callbacks for MainWindow signals to activate logging/updating functions
		self.mw.update_combos.connect(self.update_comboboxes)
		self.mw.log_now.connect(self.log_entries)

		# Empty lists to contain results filled in (or not) by annotator
		self.list_of_behavior_comboboxes = []
		self.list_of_sex_comboboxes = []
		self.list_of_species_comboboxes = []
		self.list_of_courting_partner_comboboxes = []
##
##
##
	def set_include_sex_and_species(self, include_sex, n_species, include_courting_partner):
		# Specify widget attributes that will determine which dropdown
		# categories will be rendered on the widget
		#######################################################################
		self.include_sex = include_sex
		self.n_species = n_species
		self.include_courting_partner = include_courting_partner
##
##
##
	@QtCore.pyqtSlot()
	def update_comboboxes(self):
		# Get the contents of the storage class FrameInformation's lists of behaviours
		# and set the current state of the comboboxes accordingly.
		# This function is important for:
		# 		1. Loading the state of an experiment from a saved version
		# 		2. Making the selections persistent in a trial. For example, if
		# 		   the annotater returns to a previous frame, we want the state
		# 		   of those comboboxes to be as they were left. NOT all reset and
		# 		   NOT whatever the most recent config from another frame was!
		#######################################################################
		# Check if the current frame has been saved already. If it has, extract
		# the relevant information from the FrameInformation object and store locally
		# using list comprehensions
		if self.mw.active_frame_info.saved:
			# Note that FrameInformation.behavior_list is really a list of lists
			# Each sublist is the [sex, species, behavior, and partner] list
			# for that frame. If an analysis does not include a category, the list
			# is still filled with nonetypes
			sexes = [e[0] for e in self.mw.active_frame_info.behavior_list]
			species = [e[1] for e in self.mw.active_frame_info.behavior_list]
			behaviors = [e[2] for e in self.mw.active_frame_info.behavior_list]
			courting_partner = [e[3] for e in self.mw.active_frame_info.behavior_list]

			# Iterate through indices and comboboxes in the list of behavioral
			# comboboxes
			for idx, cbb in enumerate(self.list_of_behavior_comboboxes):
				# Get the position in the combobox's list of options that the
				# stored attribute matches, save this as index. Behavior choices.
				index = cbb.findText(behaviors[idx], QtCore.Qt.MatchFixedString)
				if index >= 0:
					# If index is not 0 (default), set combobox display-item to index
					cbb.setCurrentIndex(index)

			if self.include_sex:
				for idx, scbb in enumerate(self.list_of_sex_comboboxes):
					# Get the position in the combobox's list of options that the
					# stored attribute matches, save this as index. Sex choices.
					index = scbb.findText(sexes[idx], QtCore.Qt.MatchFixedString)
					if index >= 0:
						# If index is not 0 (default), set combobox display-item to index
						scbb.setCurrentIndex(index)

			if self.n_species>1:
				for idx, spcbb in enumerate(self.list_of_species_comboboxes):
					# Get the position in the combobox's list of options that the
					# stored attribute matches, save this as index. Species choices.
					index = spcbb.findText(species[idx], QtCore.Qt.MatchFixedString)
					if index >= 0:
						# If index is not 0 (default), set combobox display-item to index
						spcbb.setCurrentIndex(index)

			if self.include_courting_partner:
				for idx, cpcbb in enumerate(self.list_of_courting_partner_comboboxes):
					# Get the position in the combobox's list of options that the
					# stored attribute matches, save this as index. Partner choices.
					index = cpcbb.findText(courting_partner[idx], QtCore.Qt.MatchFixedString)
					if index >= 0:
						# If index is not 0 (default), set combobox display-item to index
						cpcbb.setCurrentIndex(index)

		# If the curreent frame has not been saved, set all comboboxes to the
		# default values, referenced by the 0 position
		else:
			for idx, cbb in enumerate(self.list_of_behavior_comboboxes):
				cbb.setCurrentIndex(0)
				if self.include_sex:
					self.list_of_sex_comboboxes[idx].setCurrentIndex(0)
				if self.n_species > 1:
					self.list_of_species_comboboxes[idx].setCurrentIndex(0)
				if self.include_courting_partner:
					self.list_of_courting_partner_comboboxes[idx].setCurrentIndex(0)
##
##
##
	def adjust_size_widgets(self, n_individuals):
		# Re-renders and adjusts the size of the widget based on how many
		# categories and individuals are requested
		#######################################################################

		# Iterate through the count of individuals
		for i in range(n_individuals):
			# Make a new label for each
			label = QtWidgets.QLabel()
			label.setText('{}'.format(i+1))

			# Add sex box and options if desired
			if self.include_sex:
				scombo = QtWidgets.QComboBox(self)
				scombo.addItem('-')
				scombo.addItem('F')
				scombo.addItem('M')

			# Add species box and options if the desired n_species > 1
			if self.n_species > 1:
				spcombo = QtWidgets.QComboBox(self)
				spcombo.addItem('-')
				for j in range(self.n_species):
					spcombo.addItem('{}'.format(j+1))

			# Add courting partner box if desired
			if self.include_courting_partner:
				cpcombo = QtWidgets.QComboBox(self)
				cpcombo.addItem('-')

				# Choices are all other individuals except present indexed one
				for k in range(n_individuals):
					if k!=i:
						cpcombo.addItem('{}'.format(k+1))

			# Behavior box is always present in a given row. Add options
			combo = QtWidgets.QComboBox(self)
			combo.addItem("Nothing")
			combo.addItem("Courting")
			combo.addItem("Copulation")

			# Create layout and add labels and selected comboboxes
			# Also add the constructed combobox items to the attribute lists for
			# reference when frame is returned to
			hbox = QtWidgets.QHBoxLayout()
			hbox.addWidget(label)

			# Add sex combobox to row if desired
			if self.include_sex:
				hbox.addWidget(scombo)
				self.list_of_sex_comboboxes.append(scombo)

			# Add species combobox to row if desired
			if self.n_species > 1:
				hbox.addWidget(spcombo)
				self.list_of_species_comboboxes.append(spcombo)

			# Add behavior combobox to row
			hbox.addWidget(combo)

			# Add partner combobox to row if desired
			if self.include_courting_partner:
				hbox.addWidget(cpcombo)
				self.list_of_courting_partner_comboboxes.append(cpcombo)

			# Add the row element to the vertical layout
			self.vbox.addLayout(hbox)
			self.list_of_behavior_comboboxes.append(combo)

		# Define buttons and set callbacks
		self.saveButton = QtWidgets.QPushButton('Save')
		self.toggleButton = QtWidgets.QPushButton('Toggle Tracking')
		self.fixButton = QtWidgets.QPushButton('Fix Tracking')
		self.fixButton.clicked.connect(self.signal_fix_track)
		self.saveButton.clicked.connect(self.log_entries)
		self.toggleButton.clicked.connect(self.toggle_track)

		# Add buttons to bottom of layout
		spacer_label = QtWidgets.QLabel()
		spacer_label.setText('        ')
		hbox = QtWidgets.QHBoxLayout()
		hbox.addWidget(spacer_label)
		hbox.addWidget(self.fixButton)
		hbox.addWidget(self.toggleButton)
		self.vbox.addLayout(hbox)

		# Define saved indicator label and add to new horizontal layout hbox1
		self.saved_indicator = QtWidgets.QLabel()
		self.saved_indicator.resize(24,24)
		self.saved_indicator.setMaximumWidth(24)
		self.saved_indicator.setStyleSheet('background-color: red')
		hbox1 = QtWidgets.QHBoxLayout()
		hbox1.addWidget(spacer_label)
		hbox1.addWidget(self.saved_indicator)
		hbox1.addWidget(self.saveButton)
		self.vbox.addLayout(hbox1)
##
##
##
	def signal_fix_track(self):
		# Sends signal to mainwindow to dispatch track-fixing dialog
		#######################################################################
		self.fix_now.emit()
##
##
##
	def toggle_track(self):
		# Sends signal to mainwindow to request track toggle and re-render
		# from ImageWidget
		#######################################################################
		self.toggleTrack.emit()
##
##
##
	@QtCore.pyqtSlot()
	def log_entries(self):
		# Appends the 4-list of current combobox entries to the list_of_behaviors
		# list attribute and signls main window to update storage containers
		# and save.
		#######################################################################
		# Reinitialize the list of behaviors structure
		self.list_of_behaviors = []

		# Itearate through the behavior comboboxes
		for idx, i in enumerate(self.list_of_behavior_comboboxes):
			# Sex info
			if self.include_sex:
				# Append sex info
				sex = str(self.list_of_sex_comboboxes[idx].currentText())

			else:
				# No sex info, append -
				sex = '-'

			# Species info
			if self.n_species > 1:
				# Append species info
				species = str(self.list_of_species_comboboxes[idx].currentText())

			else:
				# No species info, append -
				species = '-'

			# Partner info
			if self.include_courting_partner:
				# Append partner info
				partner = str(self.list_of_courting_partner_comboboxes[idx].currentText())

			else:
				# No partner info, append -
				partner = '-'

			# Apppend behavior info
			behavior = str(i.currentText())
			self.list_of_behaviors.append([sex, species, behavior, partner])

		# Emit signals to mainwindow
		self.newListofBehaviors.emit()
		self.signal_save.emit()
##
##
##
##
class ImageWidget(QtWidgets.QWidget):
	#######################################################################
	# One of the two main widgets on the main window, this is the left-side
	# large image display. The display itself is a large label that has a
	# pixmap mapped to it.
	# Member functions are mostly to modify and set the pixmap label and
	# manage a few small peripheral features like inc/dec button and framecount
	#######################################################################
	# Define signals
	request_frame_status = QtCore.pyqtSignal(str)	# Get frame count out of total
	signal_save = QtCore.pyqtSignal()				# Trigger save routine
	def __init__(self, parent):
		# Initialize ui and layout components
		#######################################################################
		super(ImageWidget, self).__init__(parent)
		vbox = QtWidgets.QVBoxLayout()
		hbox1 = QtWidgets.QHBoxLayout()
		hbox2 = QtWidgets.QHBoxLayout()
		self.label = QtWidgets.QLabel()
		self.label.move(10, 40)
		self.label.resize(700, 700)
		hbox1.addWidget(self.label)
		self.forwardPB = QtWidgets.QPushButton('>>')
		self.backwardPB = QtWidgets.QPushButton('<<')
		self.frameLabel = QtWidgets.QLabel('                         ')
		self.frameLabel.setMaximumWidth(60)
		hbox2.addWidget(self.backwardPB)
		hbox2.addWidget(self.frameLabel)
		hbox2.addWidget(self.forwardPB)
		vbox.addLayout(hbox1)
		vbox.addLayout(hbox2)
		self.setLayout(vbox)

		# Set callbacks for buttons and signal from mainwindow
		self.forwardPB.clicked.connect(self.increment_frame_count)
		self.backwardPB.clicked.connect(self.decrement_frame_count)
		self.parent().frameTuple.connect(self.updateFrameLabel)

		# Initialize a black image and set the label to this until analysis
		blankImage = np.zeros((900,900,3),dtype=np.uint8)
		self.setLabel(blankImage)
##
##
##
	def increment_frame_count(self):
		# Send a signal to main window on forward button press to advance a frame
		#######################################################################
		self.signal_save.emit()
		self.request_frame_status.emit('inc')
##
##
##
	def decrement_frame_count(self):
		# Send a signal to main window on backward button press to reverse a frame
		#######################################################################
		self.signal_save.emit()
		self.request_frame_status.emit('dec')
##
##
##
	@QtCore.pyqtSlot(int, int)
	def updateFrameLabel(self, v1, v2):
		# Set the frame progress label
		#######################################################################
		self.frameLabel.setText('    {} / {}'.format(v1, v2))
##
##
##
	@QtCore.pyqtSlot(QtGui.QImage)
	def setLabel(self, image):
		# Set the main display label. Called every time the main display is updated
		#######################################################################
		# Convert the input image to color and perform deep copy
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		image = image.copy()

		# Make sure image does not exceed maximum size
		if image.shape[0] > 1024 or image.shape[1]>1024:
			image = cv2.resize(image, (1024,1024))

		# Cast the image into QImage type and scale to 900,900
		self.qimage = QtGui.QImage(image, image.shape[0], image.shape[1], QtGui.QImage.Format_RGB888)
		self.qimage = self.qimage.scaled(900,900, QtCore.Qt.KeepAspectRatio)

		# Cast the QImage into QPixmap type
		pixmap = QtGui.QPixmap(self.qimage)

		# Set label to show pixmap and display
		self.label.setPixmap(pixmap)
		self.label.show()
