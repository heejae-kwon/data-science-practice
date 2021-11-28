import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.gray_cat = self.get_file_path("gray_cat.jpg")
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def filter2D(self):
        image = cv2.imread(self.gray_cat)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        size = 4
        kernel = np.ones((size, size), np.float32) / (size ** 2)
        print(kernel)

        dst = cv2.filter2D(image, -1, kernel)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("blur_filter2D.jpg"), dst)
        return

    def basic_blur(self):
        image = cv2.imread(self.gray_cat)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        dst = cv2.blur(image, (4, 4))
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("blur_basic.jpg"), dst)
        return
    
    def Gaussian_blur(self):
        image = cv2.imread(self.gray_cat)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

        # kernel_size: 홀수
        dst = cv2.GaussianBlur(image, (5, 5), 0)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("blur_Gaussian.jpg"), dst)
        return




    def run(self):
        self.filter2D()
        self.basic_blur()
        self.Gaussian_blur()
        return




def main():
    img_modifier = ImageModifier()
    img_modifier.run()
    return

if __name__ == "__main__":
    main()

