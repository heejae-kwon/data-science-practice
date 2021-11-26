from typing import Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.cat_dir = self.get_file_path("cat.jpg")
        self.gray_cat_dir = self.get_file_path("gray_cat.jpg")
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def threshold_cat(self):
        image = cv2.imread(self.gray_cat_dir, cv2.IMREAD_GRAYSCALE)
        images : List[Tuple[Any, str]] = [] 
        ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        ret, thres2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thres3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        ret, thres4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        ret, thres5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
        images.append((thres1,"THRESH_BINARY"))
        images.append((thres2, "THRESH_BINARY_INV"))
        images.append((thres3,"THRESH_TRUNC"))
        images.append((thres4,"THRESH_TOZERO"))
        images.append((thres5,"THRESH_TOZERO_INV"))

        for i in images:
            plt.imshow(cv2.cvtColor(i[0], cv2.COLOR_GRAY2RGB))
            plt.show()
            cv2.imwrite(self.get_file_path(f"threshold_{i[1]}_cat.jpg"), i[0])

        return    

    def adaptive_threshold_cat(self):
        image = cv2.imread(self.gray_cat_dir, cv2.IMREAD_GRAYSCALE)

        ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        thres2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        plt.show()

        plt.imshow(cv2.cvtColor(thres1, cv2.COLOR_GRAY2RGB))
        plt.show()

        plt.imshow(cv2.cvtColor(thres2, cv2.COLOR_GRAY2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path(f"adaptive_threshold_cat.jpg"), thres2)

        return

    def run(self):
        self.threshold_cat()
        self.adaptive_threshold_cat()

        return





def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

