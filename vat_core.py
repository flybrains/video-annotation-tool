

class VideoInformation(object):
    def __init__(self, video_name):
        self.video_name = video_name
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_frame(self, index):
        return self.frame[index]

    def get_frame_list(self):
        return self.frames

class ContourPoint(object):
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class FrameInformation(object):
    def __init__(self, index, video_name, n_individuals):
        self.index = index
        self.video = video_name
        self.n_individuals = n_individuals
        self.list_of_contour_points = []
        self.neighbor_dict = {}

    def detect_contours(self):
        pass

    def log_valid_positions(self):
        for i in range(self.n_individuals):
            self.list_of_contour_points.append()

    def calculate_neighbor_distances(self):
        pass


    def get_info(self):
        pass

class InfoWriter(object):
    def __init__(self, video_name):
        self.video_name = video_name

    def _make_output_name(self):
        self.output_name = 's'
