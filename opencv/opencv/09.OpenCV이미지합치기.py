import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
        self.cat_dir = self.get_file_path("cat.jpg")
        self.bubble_dir = self.get_file_path("bubble.jpg")
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def saturation_cat(self):
        image_1 = cv2.imread(self.cat_dir)
        image_2 = cv2.imread(self.bubble_dir)

        result = cv2.add(image_1, image_2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("saturation_cat.jpg"), result)

        return    

    def modulo_cat(self):
        image_1 = cv2.imread(self.cat_dir)
        image_2 = cv2.imread(self.bubble_dir)

        result = image_1 + image_2
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(self.get_file_path("modulo_cat.jpg"), result)
        return

    def run(self):
        self.saturation_cat()
        self.modulo_cat()

        return




def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

