"""
Core information storage classes

schema:

VideoInformation
    |____metadata
    |____video name
    |____list of M FrameInformation objects
            |___FrameInformation Object 1
            |___FrameInformation Object 2
            ...
            |___FrameInformation Object M
                |___frame information
                |___list of qualitative behaviors
                |___list of contour points
                        |____Contour Point 1
                        |____Contour Point 2
                        ...
                        |____Contour Point N

"""
##
##
##
##
class VideoInformation(object):
    #######################################################################
    # Information storage class for the entire video analysis
    #######################################################################
    def __init__(self, video_name, metadata):
        self.video_name = video_name    # Address of video/data location
        self.frames = []                # List comprised of FrameInformation objects
        self.metadata = metadata        # Dictionary of various video paramaters,

    def get_frame(self, index):
        # Return the frame object from a given index of the ideo's FrameInformation object list
        #######################################################################
        return self.frame[index]
##
##
##
    def get_frame_list(self):
        # Return ideo's entire FrameInformation object list
        #######################################################################
        return self.frames
##
##
##
    def add_frame(self, frame):
        # Append a frame object to the video's FrameInformation object list
        #######################################################################
        self.frames.append(frame)
##
##
##
##
class ContourPoint(object):
    #######################################################################
    # Information storage class for the attributes of a detected contour
    # In an N-animal analysis, there will be N ContourPoint objects per frame
    #######################################################################
    def __init__(self, id, x, y, c, label_config):
        self.id = id                        # Sequentially-assigned index id (0:N)
        self.x = x                          # Detected x position in pixels
        self.y = y                          # Detected y position in pixels
        self.c = c                          # Copy of cv2's generated contour object
        self.label_config = label_config    # Spatial configuration to prevent
##
##
##
##
class FrameInformation(object):
    #######################################################################
    # Information strage class in the scope of a single frame
    # In an M-frame analysis, there will be M FrameInformation objects
    #######################################################################
    def __init__(self, index, video_name, n_individuals, frameNo):
        self.index = index                  # The index of the frame in order (0:M)
        self.video = video_name             # Address of source video
        self.n_individuals = n_individuals  # Number of animals expected per frame
        self.frameNo = frameNo              # Frame number relative to source video
        self.list_of_contour_points = []    # List of detected/corrected ContourPointObjects
        self.tracked = False                # Boolean whether frame shows tracking or not
        self.behavior_list = []             # List of user-assigned behaviors/traits
        self.saved = False                  # Boolean whether frame saved or not
##
##
##
    def add_contour_point(self, pt):
        # Append a ContourPoint object to list of contour points
        #######################################################################
        self.list_of_contour_points.append(pt)
##
##
##
    def get_list_of_contour_points(self):
        # Return List of ContourPoint objects
        #######################################################################
        return self.list_of_contour_points
