from typing import Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.cat_dir = self.get_file_path("cat.jpg")
        self.image : np.ndarray = None
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def change_color(self, x):
        r = cv2.getTrackbarPos("R", "Image")
        g = cv2.getTrackbarPos("G", "Image")
        b = cv2.getTrackbarPos("B", "Image")
        self.image[:] = [b, g, r]
        cv2.imshow('Image', self.image)
  
    def opencv_tracker(self):
        self.image = np.zeros((600, 400, 3), np.uint8)
        cv2.namedWindow("Image")

        cv2.createTrackbar("R", "Image", 0, 255, self.change_color)
        cv2.createTrackbar("G", "Image", 0, 255, self.change_color)
        cv2.createTrackbar("B", "Image", 0, 255, self.change_color)

        cv2.imshow('Image', self.image)
        cv2.waitKey(0)

        return

    def run(self):
        self.opencv_tracker()

        return





def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

