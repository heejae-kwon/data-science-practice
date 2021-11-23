import cv2
import matplotlib.pyplot as plt
import time
from pathlib import Path

def get_cat_file_path(file_name : str) -> str:
    pic_dir = str(Path(__file__).parent.resolve()) + str(Path("/pictures"))
    return pic_dir+str(Path(f"/{file_name}"))

def pixel_color_cat():
    image = cv2.imread(get_cat_file_path("cat.jpg"))
    image[:, :, 2] = 0

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(get_cat_file_path("pixel_color_cat.jpg"), image)

    return
     

def pixel_range_cat():
    image = cv2.imread(get_cat_file_path("cat.jpg"))

    start_time = time.time()
    for i in range(0, 100):
     for j in range(0, 100):
        image[i, j] = [255, 255, 255]
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    image[0:100, 0:100] = [0, 0, 0]
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(get_cat_file_path("pixel_range_cat.jpg"), image)

    return


def roi_cat():
    image = cv2.imread(get_cat_file_path("cat.jpg"))
    # Numpy Slicing: ROI 처리 가능
    roi = image[200:350, 50:200]

    # ROI 단위로 이미지 복사하기
    image[0:150, 0:150] = roi

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite(get_cat_file_path("roi_cat.jpg"), image)


    return

def main():
    pixel_range_cat()
    roi_cat()
    pixel_color_cat()

    return

if __name__ == "__main__":
    main()

