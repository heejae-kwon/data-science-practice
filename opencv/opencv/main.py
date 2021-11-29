from typing import Any, List, Tuple
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

    def run(self):
        return




def main():
    img_modifier = ImageModifier()
    img_modifier.run()
    return

if __name__ == "__main__":
    main()

