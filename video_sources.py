import cv2

class BaseVideoSource:
    def read(self): raise NotImplementedError
    def is_opened(self) -> bool: raise NotImplementedError
    def release(self): raise NotImplementedError

class WebcamSource(BaseVideoSource):
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(int(index))
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if not self.cap: return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

class IPCameraSource(BaseVideoSource):
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(str(url))

    def read(self):
        if not self.cap: return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
