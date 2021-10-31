import cv2
from utils import Singleton
from ml import ML

class Camera(Singleton):
    count = 0
    def __init__(self):
        self.width = 224
        self.height = 224
        self.capture_fps = 30
        self.capture_width = 640
        self.capture_height = 480
        self.capture_device = 0
        if Camera.count == 0:
            try:
                self.video = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
                re , image = self.video.read()
                if not re:
                    raise RuntimeError('Could not read image from camera.')
            except:
                raise RuntimeError('Could not initialize camera.  Please see error trace.')
        Camera.count += 1

    def _gst_str(self):
        return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(self.capture_device, self.capture_width, self.capture_height, self.capture_fps)

    def __del__(self):
        self.video.release()
        Camera.count -= 1

    def read(self):
        re, image = self.video.read()
        if re:
            image_resized = cv2.resize(image,(int(self.width),int(self.height)))
            return re, image_resized
        else:
            raise RuntimeError('Could not read image from camera')

    def get_frame(self):
        return True, ML().execute(self.read()[1])#, ML().draw_objects.isDone
