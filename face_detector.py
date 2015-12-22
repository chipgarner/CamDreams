import cv2


class FaceDetector:

    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier('ImagesIn/haarcascade_frontalface_default.xml')

    faceCascade = None

    def get_faces(self, frame, back_image):
        back_img = back_image.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_scale = 3
        scl = 1.0 / image_scale
        smallgray = cv2.resize(gray, (0, 0), fx=scl, fy=scl)
        # May help slightly eqsmall = cv2.equalizeHist(smallgray)

        faces = self.faceCascade.detectMultiScale(
                smallgray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        got_face = False
        for (x, y, w, h) in faces:
            got_face = True

            x1 = int(x * image_scale)
            x2 = int((x + w) * image_scale)
            y1 = int(y * image_scale)
            y2 = int((y + h) * image_scale)
            deltay = (y2 - y1) * 0.2
            y1 -= deltay
            if y1 < 0:
                y1 = 0
            y2 += deltay
            h, w = back_img.shape[:2]
            if y2 > w:
                y2 = w
            back_img[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            # cv2.rectangle(back_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return got_face, back_img
