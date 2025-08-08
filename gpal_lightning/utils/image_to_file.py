import cv2
import numpy as np


def image_to_file(path: str, image: np.ndarray):
    """This function takes in BGR and transposed image array and use cv2 to dump it"""
    image = np.transpose(image, (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)
