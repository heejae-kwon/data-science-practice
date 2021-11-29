from typing import Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path

class KNNRunner:
    def __init__(self):
        return

    def get_file_path(self, file_name : str) -> str:
        pic_dir = str(Path(__file__).parent.resolve())+str(Path(f"/{file_name}"))
        print(pic_dir)
        return pic_dir
    

    def run(self):

        return




def main():
    knn_runner = KNNRunner()
    knn_runner.run()
    return

if __name__ == "__main__":
    main()

