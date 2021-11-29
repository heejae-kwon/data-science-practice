import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.digital_image = self.get_file_path("digit_image.jpg")
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def bounding_rect(self):
        image = cv2.imread(self.digital_image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
        thresh = cv2.bitwise_not(thresh)

        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        plt.show()

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ctr = np.array(contours[1]).reshape((-1,1,2)).astype(np.int32)
        image = cv2.drawContours(image,  ctr, -1, (0, 0, 255), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("contour_digit_num.jpg"), image)

        #contour = contours[0]
        x, y, w, h = cv2.boundingRect(ctr)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("contour_digit_num_bounding.jpg"), image)

        return
    

    def convex_hull(self):
        image = cv2.imread(self.digital_image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
        thresh = cv2.bitwise_not(thresh)

        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        plt.show()

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ctr = np.array(contours[1]).reshape((-1,1,2)).astype(np.int32)
        image = cv2.drawContours(image,  ctr, -1, (0, 0, 255), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        #contour = contours[0]
        hull = cv2.convexHull(ctr)
        image = cv2.drawContours(image, [hull], -1, (255, 0, 0), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("contour_digit_num_convex_hull.jpg"), image)

        return
    

    def approx_poly_dp(self):
        image = cv2.imread(self.digital_image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
        thresh = cv2.bitwise_not(thresh)

        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        plt.show()

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.array(contours[1]).reshape((-1,1,2)).astype(np.int32)
        image = cv2.drawContours(image,  contour, -1, (0, 0, 255), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        #contour = contours[0]
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        image = cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("contour_digit_num_approx_poly_dp.jpg"), image)

        return

    def contour_calculations(self):
        image = cv2.imread(self.digital_image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 230, 255, 0)
        thresh = cv2.bitwise_not(thresh)

        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        plt.show()

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.array(contours[1]).reshape((-1,1,2)).astype(np.int32)
        image = cv2.drawContours(image,  contour, -1, (0, 0, 255), 4)

        area = cv2.contourArea(contour)
        print(area)

        length = cv2.arcLength(contour, True)
        print(length)

        M = cv2.moments(contour)
        print(M)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        return




    def run(self):
        self.bounding_rect()
        self.convex_hull()
        self.approx_poly_dp()
        self.contour_calculations()

        return




def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

