import cv2


def make_blur(img):
    ksize = (10, 10)
    blur = cv2.blur(img, ksize)
    return blur
