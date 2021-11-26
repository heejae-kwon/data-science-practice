from typing import Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class ImageModifier:
    def __init__(self):
       # self.cat_dir = self.get_file_path("cat.jpg")
       # self.image : np.ndarray = None
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
        return pic_dir+str(Path(f"/{file_name}"))

    def draw_line(self):
        image = np.full((512, 512, 3), 255, np.uint8)
        image = cv2.line(image, (0, 0), (255, 255), (255, 0, 0), 3)
        plt.imshow(image)
        plt.show()
        cv2.imwrite(self.get_file_path(f"line.jpg"), image)
        return

    def draw_rectangle(self):
        image = np.full((512, 512, 3), 255, np.uint8)
        image = cv2.rectangle(image, (20, 20), (255, 255), (255, 0, 0), 3)
        plt.imshow(image)
        plt.show()
        cv2.imwrite(self.get_file_path(f"rectangle.jpg"), image)
        return
    
    def draw_circle(self):
        image = np.full((512, 512, 3), 255, np.uint8)
        image = cv2.circle(image, (255, 255), 30, (255, 0, 0), 3)
        plt.imshow(image)
        plt.show()
        cv2.imwrite(self.get_file_path(f"circle.jpg"), image)
        return 
    
    def draw_polylines(self):
        image = np.full((512, 512, 3), 255, np.uint8)
        points = np.array([[5, 5], [128, 258], [483, 444], [400, 150]])
        image = cv2.polylines(image, [points], True, (0, 0, 255), 4)
        plt.imshow(image)
        plt.show()
        cv2.imwrite(self.get_file_path(f"polylines.jpg"), image)
        return
    
    def draw_text(self):
        image = np.full((512, 512, 3), 255, np.uint8)
        image = cv2.putText(image, 'Hello World', (0, 200), cv2.FONT_ITALIC, 2, (255, 0, 0))
        plt.imshow(image)
        plt.show()        
        cv2.imwrite(self.get_file_path(f"text.jpg"), image)
        return


    def run(self):
        self.draw_line()
        self.draw_rectangle()
        self.draw_circle()
        self.draw_polylines()
        self.draw_text()

        return





def main():
    img_modifier = ImageModifier()
    img_modifier.run()

    return

if __name__ == "__main__":
    main()

