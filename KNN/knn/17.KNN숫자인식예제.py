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
    

    def create_npz(self):
        img = cv2.imread(self.get_file_path('img/digits.png'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 세로로 50줄, 가로로 100줄로 사진을 나눕니다.
        cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
        x = np.array(cells)

        # 각 (20 X 20) 크기의 사진을 한 줄(1 X 400)으로 바꿉니다.
        train = x[:, :].reshape(-1, 400).astype(np.float32)

        # 0이 500개, 1이 500개, ... 로 총 5,000개가 들어가는 (1 x 5000) 배열을 만듭니다.
        k = np.arange(10)
        train_labels = np.repeat(k, 500)[:, np.newaxis]

        np.savez(self.get_file_path("trained.npz"), train=train, train_labels=train_labels)
        self.create_text_images(x)

        return
    
    def create_text_images(self, x:np.ndarray):
        # 다음과 같이 하나씩 글자를 출력할 수 있습니다.
        plt.imshow(cv2.cvtColor(x[0, 0], cv2.COLOR_GRAY2RGB))
        plt.show()

        # 다음과 같이 하나씩 글자를 저장할 수 있습니다.
        cv2.imwrite(self.get_file_path('img/test_0.png'), x[0, 0])
        cv2.imwrite(self.get_file_path('img/test_1.png'), x[5, 0])
        cv2.imwrite(self.get_file_path('img/test_2.png'), x[10, 0])
        cv2.imwrite(self.get_file_path('img/test_3.png'), x[15, 0])
        cv2.imwrite(self.get_file_path('img/test_4.png'), x[20, 0])
        cv2.imwrite(self.get_file_path('img/test_5.png'), x[25, 0])
        cv2.imwrite(self.get_file_path('img/test_6.png'), x[30, 0])
        cv2.imwrite(self.get_file_path('img/test_7.png'), x[35, 0])
        cv2.imwrite(self.get_file_path('img/test_8.png'), x[40, 0])
        cv2.imwrite(self.get_file_path('img/test_9.png'), x[45, 0])

        return 


    # 파일로부터 학습 데이터를 불러옵니다.
    def load_train_data(self,file_name):
        with np.load(file_name) as data:
            train = data['train']
            train_labels = data['train_labels']
        return train, train_labels
    
    # 손 글씨 이미지를 (20 x 20) 크기로 Scaling합니다.
    def resize20(self,image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resize = cv2.resize(gray, (20, 20))
        plt.imshow(cv2.cvtColor(gray_resize, cv2.COLOR_GRAY2RGB))
        plt.show()
        # 최종적으로는 (1 x 400) 크기로 반환합니다.
        return gray_resize.reshape(-1, 400).astype(np.float32)

    def check(self,test, train, train_labels):
        knn = cv2.ml.KNearest_create()
        knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
        # 가장 가까운 5개의 글자를 찾아, 어떤 숫자에 해당하는지 찾습니다.
        ret, result, neighbours, dist = knn.findNearest(test, k=5)
        return result

    def recognize_images(self):
        train, train_labels = self.load_train_data(self.get_file_path("trained.npz"))

        for file_name in glob.glob(self.get_file_path('img/test_*.png')):
            test = self.resize20(file_name)
            result = self.check(test, train, train_labels)
            print(result)

        return



    def run(self):
        self.create_npz()
        self.recognize_images()

        return




def main():
    knn_runner = KNNRunner()
    knn_runner.run()
    return

if __name__ == "__main__":
    main()

