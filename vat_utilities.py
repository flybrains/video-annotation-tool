"""
Utility classes for UI graphics and data reporting
"""
import cv2
import csv
import numpy as np
import pickle
from skimage.measure import EllipseModel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject
import vat_core as vc
import vat_ui_warnings as vuw
##
##
##
##
class PointAdder(object):
	#######################################################################
	# Class to ses mouse input to draw and record point on a given frame in a
	# new UI window.
	# Used for adding missed reads during tracking fixing and adding points
	# while selecting ellipses
	#######################################################################
	def __init__(self, frame):
		self.frame = frame
		self.point = None
##
##
##
	def add_point(self, event,x,y,flags,param):
		# On click: record point, on release: draw on recorded point coordinates
		#######################################################################
		# Define cv2 left-click callback
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix,self.iy = x,y

		# Exit on click button released
		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			cv2.circle(self.display_frame,(x,y),4,(255,255,255),-1)
			self.point = [x,y]
##
##
##
	def define_point_location(self):
		# Open a cv2 ui window and allow user to click-select where to place point
		#######################################################################
		# Initialize cv2 namedWindow
		self.display_frame = self.frame
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.add_point)

		# While a point has not been defined, display window and loop for callback
		while self.point is None:
			cv2.imshow('image',self.display_frame)
			cv2.moveWindow('image',10,10)
			# Exit on escape key
			k = cv2.waitKey(1) & 0xFF
			if k == ord('m'):
				break
			elif k == 27:
				break
		# Clean up
		cv2.destroyAllWindows()
##
##
##
	def get_point(self):
		# Return added point
		#######################################################################
		return self.point
##
##
##
##
class EllipseDrawer(object):
	#######################################################################
	# Class containing functions to make and report ellipses with mouse-click
	# Used to define chamber as well as food patches
	#######################################################################
	def __init__(self, video_address):
		# Initilaize parameters
		#######################################################################
		self.video_address = video_address
		self.drawing = False 			# Callback sets to true if mouse pressed
		self.index=1 					# May need multiple ellipses, this is indexer
		self.ix,self.iy = -1,-1
		self.points = []
		self.draw_color = (0,0,255)
		self.no_patch = False			# If no food patch, function still called
										# to provide essential info but drawing
										# step is skipped
##
##
##
	def draw_dot(self, event,x,y,flags,param):
		# Similar to above add_point but with index labelling for 5 points
		#######################################################################
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix,self.iy = x,y

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			cv2.circle(self.display_frame,(x,y),7,self.draw_color,-1)
			cv2.putText(self.display_frame,"{}/5".format(self.index),(x+10,y+10), cv2.FONT_HERSHEY_PLAIN, 1, self.draw_color)
			self.index+=1
			self.points.append([x,y])
##
##
##
	def _load_video_frame(self):
		# Grab 10th frame (10 is arbitrary, as long as no major physical shift of dish,
		# frame index does not matter)
		#######################################################################
		cap = cv2.VideoCapture(self.video_address)
		cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
		ret, self.frame = cap.read()
##
##
##
	def define_food_patches(self, n_patches):
		# Function for defining foodpatches, 0, 1 or 2 are valid args for n_patches
		# Grab a frame to draw on & display it
		#######################################################################
		self._load_video_frame()
		self.display_frame = self.frame
		cv2.namedWindow('image')

		# Define callback to draw index dots around patches
		cv2.setMouseCallback('image',self.draw_dot)

		if n_patches==0:
			# Skip routine if no food patches
			self.no_patch = True
			pass

		else:
			# While the number of defined points < 5*n_patches,
			# click, draw and log the coordinates of points
			while(len(self.points)<5*n_patches):
				if len(self.points)==5:
					self.draw_color = (255,0,0)
					self.index=1

				#Update frame with index-labelled dot and re-display
				cv2.imshow('image',self.display_frame)
				cv2.moveWindow('image',10,10)
				k = cv2.waitKey(1) & 0xFF

				if k == ord('m'):
					break
				elif k == 27:
					break

			# Return click-selected points grouped by unique food patch
			if len(self.points) == 5:
				self.segmented_points = self.points

			elif len(self.points) == 10:
				self.segmented_points = [[e for e in self.points[:5]],[e for e in self.points[5:]]]

			point_array = np.asarray(self.segmented_points)

			if n_patches==1:
				point_array = np.expand_dims(point_array, axis=0)

			# After points have been selected, draw ellipse for verification
			# If bad ellipse, should be re-selected
			colors = [(0,0,255),(255,0,0)]

			# Simultaneously create a logical 1/0 mask for later determination of food patch pixels
			self.food_patch_mask = np.zeros(self.display_frame.shape, dtype=np.uint8)

			# Init empty list for food patch centroids
			self.food_patch_centroids = []

			for i in range(n_patches):
				xy = EllipseModel()
				xy.estimate(point_array[i])
				xc,yc, a,b,theta =  int(xy.params[0]), int(xy.params[1]), int(xy.params[2]), int(xy.params[3]), int(np.rad2deg(xy.params[4]))
				self.frame = cv2.ellipse(self.display_frame, (xc,yc), (a,b), theta, 0,360,colors[i], 1)

				# Mask is same food patch drawing with filled in ellipses (note -1 argument below) in white
				# This creates a logical mask the size of frame, non-patch pixels = 0, else patch pixels = 1 (black/white)
				self.food_patch_mask = cv2.ellipse(self.food_patch_mask, (xc,yc), (a,b), theta, 0,360,(255,255,255), -1)

				# Cast the centroids into arrays and append them to centroid list
				self.food_patch_centroids.append(np.asarray([int(cx),int(cy)]))

			# Draw display frame with new ellipses
			cv2.imshow('image', self.display_frame)
			cv2.moveWindow('image',10,10)
			cv2.waitKey(0)

			# Clean up
			cv2.destroyAllWindows()
##
##
##
	def define_bowl_mask(self):
		# Similar routine to that above. Single pass for 1 chamber
		#######################################################################
		self._load_video_frame()

		# Resize total frame so all points on perimeter can be reached
		self.display_frame = cv2.resize(self.frame, None, fx=0.5, fy=0.5)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.draw_dot)

		# While the number of defined points < 5*n_patches,
		# click, draw and log the coordinates of points
		while(len(self.points)<5):
			cv2.imshow('image',self.display_frame)
			cv2.moveWindow('image',10,10)
			k = cv2.waitKey(1) & 0xFF

			if k == ord('m'):
				break
			elif k == 27:
				break

		# Create ellipse model from 5 click-selected points
		point_array = np.asarray(self.points)
		xy = EllipseModel()
		xy.estimate(point_array)
		xc,yc, a,b,theta =  int(xy.params[0]), int(xy.params[1]), int(xy.params[2]), int(xy.params[3]), int(np.rad2deg(xy.params[4]))
		display_bowl_mask = np.zeros(self.display_frame.shape, dtype=np.uint8)
		bowl_mask = np.zeros(self.frame.shape, dtype=np.uint8)

		# Create binary mask by drawing ellipse and filling in with white on black background
		display_bowl_mask = cv2.ellipse(display_bowl_mask, (xc,yc), (a,b), theta, 0,360,(255,255,255), -1)
		self.bowl_mask = cv2.ellipse(bowl_mask, (2*xc,2*yc), (2*a,2*b), theta, 0,360,(255,255,255), -1)
		self.mask_centroid = (2*xc, 2*yc)

		# Mask a video frame using generated mask
		display_masked = cv2.bitwise_and(self.display_frame, display_bowl_mask)
		masked = cv2.bitwise_and(self.frame, bowl_mask)

		# On the masked frame, draw the click-selected ellipse points and contour
		for i in range(5):
			x,y = self.points[i][0], self.points[i][1]
			cv2.circle(display_masked,(x,y),7,(0,0,255),-1)
			cv2.putText(display_masked,"{}/5".format(i+1),(x+10,y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

		# Display masked and overlaid frame
		cv2.imshow('image', display_masked)
		cv2.moveWindow('image',10,10)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
##
##
##
	def get_bowl_mask(self):
		# Return bowl mask and centroid in pixels
		#######################################################################
		return self.bowl_mask, self.mask_centroid
##
##
##
	def get_food_patches(self):
		# Return food patches mask
		#######################################################################
		if not self.no_patch:
			return self.food_patch_mask, self.food_patch_centroids

		else:
			return None, None
##
##
##
##
class Thresholder(object):
	#######################################################################
	# User interface to adjust threshold before tracking and annotation
	# Default values have been set to work for most well-lit videos,
	# However slider bars require user to make sure they work for each video
	#######################################################################
	def __init__(self, video_address, bg, center, mask):
		# Grab an arbitrary frame after initial camera start up
		# Sometimes camera adjusts exposure/params on initial startup, want to avoid
		cap = cv2.VideoCapture(video_address)
		cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
		_, self.read =cap.read()

		# Read in frame mask and computed background
		self.mask = mask
		self.bg = bg

		# Make copy of original frame
		self.og_frame = self.read.astype(np.uint8)

		# Apply mask and background subtraction
		height = self.og_frame.shape[0]
		xl, xr = int((center-(height/2))), int((center+(height/2)))
		self.og_frame = self.og_frame[:,xl:xr,:]
		self.mask = self.mask[:,xl:xr,:]
		self.gray_mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
		self.display_frame = cv2.resize(self.og_frame, None, fx=0.5, fy=0.5)
##
##
##
	def nothing(self,x):
		# Essential to have a callback for sliders, even if it does nothing
		#######################################################################
		pass
##
##
##
	def make_display_frame(self):
		# Applies tracking and overlays to frame so user can determine
		# how well tracking is working with given filter parameters
		#######################################################################
		# Copy display frame
		self.display_frame = self.og_frame.copy()
		self.display_frame = cv2.bitwise_and(self.display_frame, self.mask)

		# Convert color and threshold to put in proper format for contour detection
		self.imagem = cv2.cvtColor(self.og_frame, cv2.COLOR_BGR2GRAY)
		self.imagem=self.imagem-self.bg + 10
		ret,self.threshed_frame = cv2.threshold(self.imagem,self.thresh,255,cv2.THRESH_BINARY)
		self.imagem = cv2.bitwise_not(self.threshed_frame)
		self.imagem = cv2.bitwise_and(self.imagem, self.gray_mask)

		# Run cv2 contour detection algorithm
		self.contours, hier = cv2.findContours(self.imagem.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


		def _get(cnt):
			# Generate and return multiple statistics about each detected contour
			#######################################################################
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

		# Filter out detected contours according to their ranges in the above-calculated parameters
		# Default ranges here were determined empirically with well-lit fly bowl videos
		# These ranges are adjusted by movement of the UI sliders
		self.valids = [cnt for cnt in self.contours if (_get(cnt)[3] > self.small) and (_get(cnt)[3] < self.large) and _get(cnt)[0] < self.aspect and _get(cnt)[1] < self.extent and _get(cnt)[2] > self.solidity and _get(cnt)[-1]<self.arc]

		# Draw red edges to indicate valid contours, allow user to adjust threshold parameters
		# Should detect only fly contours, no image artifacts or shadow
		self.imagem = cv2.cvtColor(self.imagem, cv2.COLOR_GRAY2BGR)
		cv2.drawContours(self.display_frame, self.valids, -1, (0, 0, 255), 1)

		# Display results of filtering based on current thresholds
		self.display_frame = cv2.resize(self.display_frame, None, fx=0.5, fy=0.5)
##
##
##
	def get_values(self):
		# Return current values of UI sliders in threshold window
		#######################################################################
		cv2.namedWindow('Threshold')
		cv2.createTrackbar('Background','Threshold',0,255,self.nothing)
		cv2.createTrackbar('Size: Small','Threshold',0,500,self.nothing)
		cv2.createTrackbar('Size: Large','Threshold',0,500,self.nothing)
		cv2.createTrackbar('Solidity','Threshold',0,1000,self.nothing)
		cv2.createTrackbar('Extent','Threshold',0,1000,self.nothing)
		cv2.createTrackbar('Aspect','Threshold',0,1000,self.nothing)
		cv2.createTrackbar('Arc','Threshold',0,1000,self.nothing)

		cv2.setTrackbarPos('Background','Threshold',130)
		cv2.setTrackbarPos('Size: Small','Threshold',60)
		cv2.setTrackbarPos('Size: Large','Threshold',400)
		cv2.setTrackbarPos('Solidity','Threshold',58)
		cv2.setTrackbarPos('Extent','Threshold',162)
		cv2.setTrackbarPos('Aspect','Threshold',276)
		cv2.setTrackbarPos('Arc','Threshold',59)

		# While the user has not terminated the session, regenerate the display image with contour detection
		# filtered according to the ranges defined in the sliders
		while(1):
			# Show the frame and wait
			cv2.imshow('Threshold',self.display_frame)
			cv2.moveWindow('image',10,10)
			k = cv2.waitKey(1) & 0xFF
			if k == 27 or k==13:
				break

			# get current positions of four trackbars
			self.thresh = cv2.getTrackbarPos('Background','Threshold')
			self.small = cv2.getTrackbarPos('Size: Small','Threshold')
			self.large = cv2.getTrackbarPos('Size: Large','Threshold')
			self.solidity = cv2.getTrackbarPos('Solidity','Threshold')/100
			self.extent = cv2.getTrackbarPos('Extent','Threshold')/100
			self.aspect = cv2.getTrackbarPos('Aspect','Threshold')/100
			self.arc = cv2.getTrackbarPos('Arc','Threshold')/100

			self.make_display_frame()

		cv2.destroyAllWindows()

		# When session is terminated, return the slider parameters deemed sufficient
		# by user, to be stored and used in contour detection on each future frame
		return self.thresh, self.small, self.large, self.solidity, self.extent, self.aspect, self.arc
##
##
##
##
class DataWriter(object):
	#######################################################################
	# Handles all writing of data to output csv
	#######################################################################
	def __init__(self,pickle_name):
		# Load in data stored in pickle serial object
		#######################################################################
		self.pickle_name = pickle_name
		self.load_pickle()

		# A key for food patches should only be written once if needed at all,
		# this will be set to true if one is written to preven subsequent writing
		self.patch_key_made = False

	def load_pickle(self):
		# Load pickle containing vat_core.VideoInformation object
		#######################################################################
		with open(self.pickle_name, 'rb') as f:
			self.videoinfo = pickle.load(f)
##
##
##
	def _get_patch_info(self,patch, x,y):
		# Function to extract information about location and shape of food patches,
		# if any exist. Will return status (on food patch or off),
		# distance to the nearest patch and the ID of that patch
		#######################################################################
		if patch is None:
			return False, None, None

		# Check if the requested x,y point is white (255, on a patch) or not
		elif patch[int(y),int(x)]==255:
			status = True
		else:
			status = False

		# Using the requested x,y as the hinge, calculate scalar distance to
		# centroids of the food patches. Find the minimum and grab the index.
		# This is the patch closest to the given animal
		centroids = self.videoinfo.metadata['food_patch_centroids']
		hinge = np.asarray([int(x), int(y)])
		dists = [np.linalg.norm(hinge-c) for c in centroids]
		dist = min(dists)
		patch_id = dists.index(dist)
		patch_id += 1

		return status, dist, patch_id
##
##
##
	def _get_sorted_dists(self, frame_info,target_id):
		# Get the closest distances and ids in descending order from individual
		#######################################################################
		sorted_dists = []
		sorted_ids = []

		# for the given animal, set its ContourPoint information to be the target
		# the x,y attributes of the ContourPoint object will be the hinge
		target = frame_info.list_of_contour_points[target_id-1]
		hinge_point = np.asarray([int(target.x), int(target.y)])

		# Make a list of all contour points in the frame except for target
		all_but = [pt for pt in frame_info.list_of_contour_points if pt.id!=target_id]

		object_dist_tuples = []

		# For all points in the target-exclusive list, calculate distance to target
		for pt in all_but:
			to_calc = hinge_point - np.asarray([int(pt.x), int(pt.y)])
			dist = np.linalg.norm(to_calc)

			# Create a 2-tuple of the points ID and its distance to the target
			odt = (pt.id, dist)

			# Append this object-distance-tuple to a list
			object_dist_tuples.append(odt)

		# When done, sort the list of object-distance tuples using distance as a key
		# via the lambda function
		s_object_dist_tuples = sorted(object_dist_tuples, key=lambda d: d[1])

		# unweave the tuple to a sorted distances list and a CORRESPONDING sorted id list
		sorted_dists = [e[1] for e in s_object_dist_tuples]
		sorted_ids = [e[0] for e in s_object_dist_tuples]

		# Get the difference between number of allocated spots for distance info,
		# and how many are needed for this frame. Pad the diffference w. nonetypes
		# to preserve proper columning in csvmax_len
		n_to_add = self.max_len - len(sorted_ids)
		if n_to_add>1:
			for i in range(n_to_add):
				sorted_dists.append(None)
				sorted_ids.append(None)

		return sorted_dists, sorted_ids
##
##
##
	def _save_food_patch_key(self, patch):
		# A key image will be generated to help the user understand which patch is 1
	    # and which is 2
		#######################################################################
		if patch is not None:
			address = self.pickle_name.split('.')[0]+'_patch_key.jpg'
			patch_key = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
			for idx,c in enumerate(self.videoinfo.metadata['food_patch_centroids']):
				cv2.putText(patch_key,"{}".format(idx+1),(c[0],c[1]), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0), thickness=3)
			cv2.imwrite(address, patch_key)
##
##
##
	def _scan_for_longest_dist_tuple(self, videoinfo):
		# Scan all FrameInformation objects and determine how many contour points are in each
		# This number -1 will be the number of spots in our information row that we write,
		# that we must allocate for information regarding distances to all other animals
		#######################################################################
		return max([len(e.list_of_contour_points) for e in videoinfo.get_frame_list()])
##
##
##
	def _get_dish_width(self):
		# Returns the pixel width of the widest part of the dish contour mask
		#######################################################################
		max_spread = 0
		for row in range(int(self.bowl_mask.shape[0]/2)):
			# Iterate through half of the rows
			try:
				# For efficiency check every other row, will make negligible difference
				# Grab the row at current index
				row_only = self.bowl_mask[row*2,:]

				# Get positions in row with white pixel occupants
				where_white = list(np.greater(row_only, 0))

				# Get the within-row index of first white pixel
				first = where_white.index(True)

				# Reverse the list and do the same
				f_where_white =  [e for e in reversed(where_white)]
				last = len(where_white) - f_where_white.index(True)

				# Find the spread between first and last white pixel as proxy
				# for chamber width at a given row index
				spread = last - first

				# Iterate and update maximum, this will be chamber width in pixels
				if spread > max_spread:
					max_spread = spread

			except ValueError:
				pass

		self.dish_width = max_spread
##
##
##
	def make_rows(self):
		# Get the number of placeholders necessary for the distance to all others
		# portion of the data
		#######################################################################
		self.max_len = self._scan_for_longest_dist_tuple(self.videoinfo)

		# Load in necessary metadata elements
		self.foodpatch_mask = self.videoinfo.metadata['food_patch_mask']
		self.diameter_mm = self.videoinfo.metadata['chamber_d']
		self.bowl_mask = cv2.cvtColor(self.videoinfo.metadata['bowl_mask'], cv2.COLOR_BGR2GRAY)
		self._get_dish_width()

		# Get converrsion factor from user-provided dish-width
		self.conversion_factor = float(self.diameter_mm/self.dish_width)

		# Initialize empty list for construction of data csv
		self.rows = []
		if self.foodpatch_mask is not None:
			self.foodpatch_mask = cv2.cvtColor(self.foodpatch_mask, cv2.COLOR_BGR2GRAY)

		if self.videoinfo.metadata['n_patches'] > 1 and not self.patch_key_made:
			self._save_food_patch_key(self.foodpatch_mask)
			self.patch_key_made = True

		# Iterate through FrameInformation objects belonging to current VideoInformation object
		for frame_info in self.videoinfo.get_frame_list():

			# Get within-analysis frame index and within source-video index
			frame_idx = frame_info.index
			video_idx = frame_info.frameNo

			# For each FrameInformation object, iterate through ContourPoint objects,
			# Each of which corresponding to an individual within the frame
			for idx, pt in enumerate(frame_info.list_of_contour_points):
				# Extract ContourPoint spatial information and correspoing animal's
				# user-annotated behavior information.
				# This data is stored as part of ContourPoint object
				id, x, y = pt.id, pt.x, pt.y

				# This data is stored in FrameInformation objects behavior_list structure,
				# which is populated using user-facing dropdowns
				sex = frame_info.behavior_list[idx][0]
				behavior = frame_info.behavior_list[idx][2]
				species = frame_info.behavior_list[idx][1]
				courting_partner = frame_info.behavior_list[idx][3]

				# Add all relevant information to the data row for the given animal in the given frame
				row = [frame_idx, video_idx, id, sex, species, behavior, courting_partner]

				# Invoke function to get information about food patch status
				patch_status, dist_to_closest_patch, patch_id = self._get_patch_info(self.foodpatch_mask,x,y)

				# Find the closest animals to each target andimal and their corresponding distances
				sorted_dists, sorted_ids = self._get_sorted_dists(frame_info, id)

				# Add on/off patch status to data row
				row.append(patch_status)

				# If an mm conversion factor has been provided, use it and add position information to data row
				try:
					row.append(dist_to_closest_patch*self.conversion_factor)
					row.append(patch_id)
					row.append(x*self.conversion_factor)
					row.append(y*self.conversion_factor)

				except TypeError:
					row.append(dist_to_closest_patch)
					row.append(x)
					row.append(y)

				# If an mm conversion factor has been provided, use it and add relative positions to data row
				for i in range(len(sorted_dists)):
					try:
						row.append(sorted_dists[i]*self.conversion_factor)
						row.append(sorted_ids[i])

					except TypeError:
						row.append(sorted_dists[i])
						row.append(sorted_ids[i])

				# Now that all relevant data has been added to row, append it to our rows structure
				# while it waits to be written to csv
				self.rows.append(row)

		# Create a header to specify columns
		self.header = ['labelled_frame', 'video_frame', 'animal_id', "sex", "species", "behavior", "courting_partner", 'on_patch', 'dist_to_closest_patch_centroid','patch_id','x_pos','y_pos']

		# Fill in header irrespective of how many animals are present (can be many)
		for i in range(len(sorted_dists)):
			if i==0:
				self.header.append('dist_to_1st_closest')
				self.header.append('1st_closest_id')

			elif i==1:
				self.header.append('dist_to_2nd_closest')
				self.header.append('2nd_closest_id')

			elif i==2:
				self.header.append('dist_to_3rd_closest')
				self.header.append('3rd_closest_id')

			else:
				self.header.append('dist_to_{}th_closest'.format(i+1))
				self.header.append('{}th_closest_id'.format(i+1))
##
##
##
	def write_csv(self):
		# Output writer
		#######################################################################
		# Modify our data location name to be a csv address
		address = self.pickle_name.split('.')[0]+'.csv'

		# Open csv and write the above-constructed rows structure to the output data csv
		with open(address, 'w') as f:
			mywriter = csv.writer(f, delimiter=',')
			mywriter.writerow(self.header)
			for i in range(len(self.rows)):
				mywriter.writerow(self.rows[i])
