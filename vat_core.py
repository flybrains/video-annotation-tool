

class VideoInformation(object):
    def __init__(self, video_name, metadata):
        self.video_name = video_name
        self.frames = []
        self.metadata = metadata

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_frame(self, index):
        return self.frame[index]

    def get_frame_list(self):
        return self.frames

class ContourPoint(object):
    def __init__(self, id, x, y, c, label_config):
        self.id = id
        self.x = x
        self.y = y
        self.c = c
        self.label_config = label_config

class FrameInformation(object):
    def __init__(self, index, video_name, n_individuals, frameNo):
        self.index = index
        self.video = video_name
        self.n_individuals = n_individuals
        self.frameNo = frameNo
        self.list_of_contour_points = []
        self.tracked = False
        self.behavior_list = []
        self.saved = False

    def add_contour_point(self, pt):
        self.list_of_contour_points.append(pt)

    def get_list_of_contour_points(self):
        return self.list_of_contour_points


class InfoWriter(object):
    def __init__(self, video_name):
        self.video_name = video_name

    def _make_output_name(self):
        self.output_name = 's'
