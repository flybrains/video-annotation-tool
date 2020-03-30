"""
Provides ui functionality and functions to operate on main storage classes to
manually correct erroneous tracking results
"""
import cv2
import PyQt5.QtCore as QtCore
import PyQt5.QtWidgets as QtWidgets
import vat_core as vc
import vat_utilities as vu
##
##
##
##
class FixTrackingWindow(QtWidgets.QDialog):
	#######################################################################
	# interface for user to call tracking-fixing functions
	#######################################################################
	# Define action-specification signals
	split_read = QtCore.pyqtSignal(int)
	remove_read = QtCore.pyqtSignal(int)
	add_read = QtCore.pyqtSignal()

	def __init__(self,parent):
		# Make dialog window and assign callbacks
		#######################################################################
		super(FixTrackingWindow, self).__init__(parent)

		# Create buttons and textedits, add them to window layout
		self.pb1 = QtWidgets.QPushButton("Split Joined Read")
		self.pb2 = QtWidgets.QPushButton("Remove False Read")
		self.pb3 = QtWidgets.QPushButton('Add Missed Read')

		self.pb1.setMaximumWidth(140)
		self.pb2.setMaximumWidth(140)
		self.pb3.setMaximumWidth(140)

		self.sp1 = QtWidgets.QSpinBox()
		self.sp2 = QtWidgets.QSpinBox()
		self.sp1.setMaximumWidth(54)
		self.sp2.setMaximumWidth(54)

		self.donepb = QtWidgets.QPushButton('Done')

		self.grid = QtWidgets.QGridLayout()
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

		# set callbacks for the buttons
		self.pb1.clicked.connect(self.split_reads)
		self.pb2.clicked.connect(self.remove_reads)
		self.pb3.clicked.connect(self.add_reads)
		self.donepb.clicked.connect(self.close)
##
##
##
	def split_reads(self):
		# Takes the index of a ContourPoint as input and creates a second one with identical charactersitics
		# and index equal to the original ContourPoint's index + 1
		# Used to resolve when mounting or close-courting animals are joined into one read
		#######################################################################
		if int(self.sp1.value())==0:
			pass
		else:
			self.read_to_split = int(self.sp1.value())-1
			self.sp1.setValue(0)

			# Invoke log_entries to save state of dropdowns
			self.parent().log_now.emit()

			# Get the highest index present in the tracking area
			self.highest_index = len(self.parent().active_frame_info.list_of_contour_points)

			# Grab the ContourPoint object that must be split and store as self.rts (read to split)
			self.parent().rts = self.parent().active_frame_info.list_of_contour_points[self.read_to_split]

			# If splitting will not result in too many reads, perform split
			if self.highest_index < self.parent().n_animals:
				# Instantiate new ContourPoint object
				new_contour_pt = vc.ContourPoint(self.highest_index+1, self.parent().rts.x,  self.parent().rts.y,  self.parent().rts.c, self.parent().rts.label_config+1)

				# Add new point to FrameInformation object's list and update ImageWidget
				self.parent().active_frame_info.add_contour_point(new_contour_pt)
				self.parent().update_raw_image(True)

			else:
				# If splitting will  result in too many reads, throw error
				msg = 'Splitting read will exceed number of individuals, revise tracking before splitting'
				self.parent().error = vuw.ErrorMsg(msg)
				self.parent().error.show()
##
##
##
	def remove_reads(self):
		# Removes an erroneous read as specified by the user,
		# resaves/reindexes remaining reads
		#######################################################################
		if int(self.sp2.value())==0:
			pass
		else:
			self.read_to_remove = int(self.sp2.value())
			self.sp2.setValue(0)

			# Invoke log_entries to save current state of dropdowns
			self.parent().log_now.emit()

			# Check to make sure there is at least 1 point to remove
			if len(self.parent().active_frame_info.list_of_contour_points) > 0:

				# Create a new list of all points except point to be removed, reassign this
				# list to the FrameInformation object, reindex for display indices, update.
				temp = [e for e in self.parent().active_frame_info.list_of_contour_points if e.id!=self.read_to_remove]
				self.parent().active_frame_info.list_of_contour_points = temp
				for idx, e in enumerate(self.parent().active_frame_info.list_of_contour_points):
					e.id = idx+1
				self.parent().update_raw_image(True)

			else:
				# If there are not enough reads to remove, throw error
				msg = 'No reads to remove'
				self.parent().error = vuw.ErrorMsg(msg)
				self.parent().error.show()
##
##
##
	def add_reads(self):
		# Creates a UI window using the vat_utilities PointAdder class and allows user
		# to click the location of a new point to be saved as a ContourPoint
		# This is used to add a missed read to the tracking.
		#######################################################################
		# Invoke log_entries to save state of dropdowns
		self.parent().log_now.emit()

		# Make sure that there is not already a full set of animals in the frame
		if len(self.parent().active_frame_info.list_of_contour_points) < self.parent().n_animals:

			# Make a smaller window for click-Selecting
			fitted = cv2.resize(self.parent().currentRawImage, None, fx=0.5, fy=0.5)

			# Invoke vat_utilities PointAdder class and allow user to pick point for new animal
			self.point_adder = vu.PointAdder(fitted)
			self.point_adder.define_point_location()
			point = self.point_adder.get_point()

			# Get the highest id number of present animal in the frame
			self.highest_index = len(self.parent().active_frame_info.list_of_contour_points)

			# Make a new ContourPoint object using the highest_index+1, and 2 X the click-selected
			# x/y coordinates (remember we scaled window by half). Add the point to the FrameInformation object
			new_contour_pt = vc.ContourPoint(self.highest_index+1, int(2*point[0]),  int(2*point[1]), None, 1)
			self.parent().active_frame_info.add_contour_point(new_contour_pt)

			# Resave ContourPoint objects as list with proper indices
			self.parent().active_frame_info.list_of_contour_points = [e for e in self.parent().active_frame_info.list_of_contour_points]

			# Update the display indices (we want these 1-indexed on the frame, not 0), and update frame
			for idx, e in enumerate(self.parent().active_frame_info.list_of_contour_points):
				e.id = idx+1
			self.parent().update_raw_image(True)

		else:
			# If there is already a full set of animals in the frame, throw an error
			msg = 'Adding read will exceed number of individuals, revise tracking before splitting'
			self.parent().error = vuw.ErrorMsg(msg)
			self.parent().error.show()
