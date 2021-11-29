import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.cat_dir = self.get_file_path("digit_image.jpg")
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def find_contours(self):
        image = cv2.imread(self.cat_dir)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0)

        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        plt.show()

        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #fix the origin code
        image = cv2.drawContours(image, [contours[1]], -1, (0, 255, 0), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("find_contours.jpg"), image)

        return


    def run(self):
        self.find_contours()

        return




def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

