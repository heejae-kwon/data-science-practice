import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_cat_file_path(file_name : str) -> str:
    pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
    return pic_dir+str(Path(f"/{file_name}"))


cat_dir = get_cat_file_path("cat.jpg")


def resize_cat():
    image = cv2.imread(cat_dir)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    expand = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    plt.imshow(cv2.cvtColor(expand, cv2.COLOR_BGR2RGB))
    plt.show()

    shrink = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    plt.imshow(cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB))
    plt.show()

    return


def move_cat():
    image = cv2.imread(cat_dir)

    # 행과 열 정보만 저장합니다.
    height, width = image.shape[:2]

    # x 50, y 10
    M = np.float32([[1, 0, 50], [0, 1, 10]])
    dst = cv2.warpAffine(image, M, (width, height))

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(get_cat_file_path("move_cat.jpg"),dst)

    return

def rotate_cat():
    image = cv2.imread(cat_dir)

    # 행과 열 정보만 저장합니다.
    height, width = image.shape[:2]

    M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 0.5)
    dst = cv2.warpAffine(image, M, (width, height))

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(get_cat_file_path("rotate_cat.jpg"),dst)

    return

def main():
    resize_cat()
    move_cat()
    rotate_cat()


    return

if __name__ == "__main__":
    main()

