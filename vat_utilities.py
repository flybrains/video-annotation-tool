import cv2
import numpy as np
from skimage.measure import EllipseModel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject

class PointAdder(object):
	def __init__(self, frame):
		self.frame = frame
		self.point = None

	def add_point(self, event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix,self.iy = x,y

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			cv2.circle(self.display_frame,(x,y),4,(255,255,255),-1)
			self.point = [x,y]

	def define_point_location(self):
		self.display_frame = self.frame
		#self.display_frame = cv2.resize(self.frame, None, fx=0.5, fy=0.)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.add_point)

		while self.point is None:
			cv2.imshow('image',self.display_frame)
			cv2.moveWindow('image',10,10)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('m'):
				break
			elif k == 27:
				break
		cv2.destroyAllWindows()

	def get_point(self):
		return self.point

class EllipseDrawer(object):
	def __init__(self, video_address):
		self.video_address = video_address
		self.drawing = False # true if mouse is pressed
		self.index=1 # if True, draw rectangle. Press 'm' to toggle to curve
		self.ix,self.iy = -1,-1
		self.points = []
		self.draw_color = (0,0,255)

	def draw_dot(self, event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.ix,self.iy = x,y

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			cv2.circle(self.display_frame,(x,y),7,self.draw_color,-1)
			cv2.putText(self.display_frame,"{}/5".format(self.index),(x+10,y+10), cv2.FONT_HERSHEY_PLAIN, 1, self.draw_color)
			self.index+=1
			self.points.append([x,y])

	def _load_video_frame(self):
		cap = cv2.VideoCapture(self.video_address)
		frame_no = 1000
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
		ret, self.frame = cap.read()

	def define_food_patches(self, n_patches):
		self._load_video_frame()
		self.display_frame = self.frame
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.draw_dot)

		if n_patches==0:
			self.no_patch = True
			pass
		else:
			while(len(self.points)<5*n_patches):
				if len(self.points)==5:
					self.draw_color = (255,0,0)
					self.index=1
				cv2.imshow('image',self.display_frame)
				cv2.moveWindow('image',10,10)
				k = cv2.waitKey(1) & 0xFF
				if k == ord('m'):
					break
				elif k == 27:
					break

			if len(self.points) == 5:
				self.segmented_points = self.points
			elif len(self.points) == 10:
				self.segmented_points = [[e for e in self.points[:5]],[e for e in self.points[5:]]]

			point_array = np.asarray(self.segmented_points)

			if n_patches==1:
				point_array = np.expand_dims(point_array, axis=0)

			colors = [(0,0,255),(255,0,0)]
			self.food_patch_mask = np.zeros(self.display_frame.shape, dtype=np.uint8)
			for i in range(n_patches):
				xy = EllipseModel()
				xy.estimate(point_array[i])
				xc,yc, a,b,theta =  int(xy.params[0]), int(xy.params[1]), int(xy.params[2]), int(xy.params[3]), int(np.rad2deg(xy.params[4]))
				self.frame = cv2.ellipse(self.display_frame, (xc,yc), (a,b), theta, 0,360,colors[i], 1)
				self.food_patch_mask = cv2.ellipse(self.food_patch_mask, (xc,yc), (a,b), theta, 0,360,(255,255,255), -1)
			cv2.imshow('image', self.display_frame)
			cv2.moveWindow('image',10,10)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

	def define_bowl_mask(self):
		self._load_video_frame()
		self.display_frame = cv2.resize(self.frame, None, fx=0.5, fy=0.5)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.draw_dot)

		while(len(self.points)<5):
			cv2.imshow('image',self.display_frame)
			cv2.moveWindow('image',10,10)
			k = cv2.waitKey(1) & 0xFF
			if k == ord('m'):
				break
			elif k == 27:
				break

		point_array = np.asarray(self.points)
		xy = EllipseModel()
		xy.estimate(point_array)
		xc,yc, a,b,theta =  int(xy.params[0]), int(xy.params[1]), int(xy.params[2]), int(xy.params[3]), int(np.rad2deg(xy.params[4]))
		display_bowl_mask = np.zeros(self.display_frame.shape, dtype=np.uint8)
		bowl_mask = np.zeros(self.frame.shape, dtype=np.uint8)

		display_bowl_mask = cv2.ellipse(display_bowl_mask, (xc,yc), (a,b), theta, 0,360,(255,255,255), -1)
		self.bowl_mask = cv2.ellipse(bowl_mask, (2*xc,2*yc), (2*a,2*b), theta, 0,360,(255,255,255), -1)
		self.mask_centroid = (2*xc, 2*yc)

		display_masked = cv2.bitwise_and(self.display_frame, display_bowl_mask)
		masked = cv2.bitwise_and(self.frame, bowl_mask)

		np.save('/home/patrick/Desktop/mask.npy',self.bowl_mask)

		for i in range(5):
			x,y = self.points[i][0], self.points[i][1]
			cv2.circle(display_masked,(x,y),7,(0,0,255),-1)
			cv2.putText(display_masked,"{}/5".format(i+1),(x+10,y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

		cv2.imshow('image', display_masked)
		cv2.moveWindow('image',10,10)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def get_bowl_mask(self):
		return self.bowl_mask, self.mask_centroid

	def get_food_patches(self):
		if not self.no_patch:
			return self.food_patch_mask
		else:
			return None

class Thresholder(object):
	def __init__(self, video_address, bg, center, mask):
		cap = cv2.VideoCapture(video_address)
		cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
		_, self.read =cap.read()

		self.mask = mask
		self.bg = bg
		self.og_frame = self.read.astype(np.uint8)

		height = self.og_frame.shape[0]
		xl, xr = int((center-(height/2))), int((center+(height/2)))
		self.og_frame = self.og_frame[:,xl:xr,:]
		self.mask = self.mask[:,xl:xr,:]
		self.gray_mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
		self.display_frame = cv2.resize(self.og_frame, None, fx=0.5, fy=0.5)

	def nothing(self,x):
		pass

	def make_display_frame(self):

		self.display_frame = self.og_frame.copy()
		self.display_frame = cv2.bitwise_and(self.display_frame, self.mask)


		self.imagem = cv2.cvtColor(self.og_frame, cv2.COLOR_BGR2GRAY)
		self.imagem=self.imagem-self.bg + 10

		ret,self.threshed_frame = cv2.threshold(self.imagem,self.thresh,255,cv2.THRESH_BINARY)
		self.imagem = cv2.bitwise_not(self.threshed_frame)
		self.imagem = cv2.bitwise_and(self.imagem, self.gray_mask)

		self.contours, hier = cv2.findContours(self.imagem.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

		self.valids = [cnt for cnt in self.contours if (_get(cnt)[3] > self.small) and (_get(cnt)[3] < self.large) and _get(cnt)[0] < self.aspect and _get(cnt)[1] < self.extent and _get(cnt)[2] > self.solidity and _get(cnt)[-1]<self.arc]
		#self.valids = [contour for contour in self.contours if (cv2.contourArea(contour) > self.small) and (cv2.contourArea(contour) < self.large)]# and (cv2.arcLength(contour,True)/cv2.contourArea(contour)) < 0.8 ]

		self.imagem = cv2.cvtColor(self.imagem, cv2.COLOR_GRAY2BGR)
		cv2.drawContours(self.display_frame, self.valids, -1, (0, 0, 255), 1)
		self.display_frame = cv2.resize(self.display_frame, None, fx=0.5, fy=0.5)


	def get_values(self):
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


		while(1):
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
		return self.thresh, self.small, self.large, self.solidity, self.extent, self.aspect, self.arc


# if __name__=="__main__":
# 	video_address = '/home/patrick/code/video-annotation-tool/sample/sample1.mp4'
# 	bg = np.load('/home/patrick/Desktop/bg.npy')
# 	mask = np.load('/home/patrick/Desktop/mask.npy')
# 	center =948
# 	th = Thresholder(video_address, bg, center, mask)
# 	th.get_values()
