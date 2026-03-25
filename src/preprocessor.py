import cv2
import numpy as np
import random


class ImageProcessor:
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            pad = int(w * 0.15)
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)

            roi = gray[y1:y2, x1:x2]
            return cv2.resize(roi, (self.img_size, self.img_size))
        return None

    def apply_augmentation(self, image):
        img_aug = image.copy()
        if random.random() > 0.5: img_aug = cv2.flip(img_aug, 1)
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-10, 10)
        return cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)