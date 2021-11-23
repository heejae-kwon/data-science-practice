import pathlib
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
    cat_jpg = pic_dir+str(Path("/cat.jpg"))
    print(cat_jpg)
    # load basic picture
    img_basic = cv2.imread(cat_jpg, cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(img_basic, cv2.COLOR_BGR2RGB))
    plt.show()

    # make it to gray
    img_basic = cv2.cvtColor(img_basic, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(img_basic, cv2.COLOR_GRAY2RGB))
    plt.show()
    gray_cat = pic_dir+str(Path("/gray_cat.jpg"))
    cv2.imwrite(gray_cat,img_basic)


    return

if __name__ == "__main__":
    main()

