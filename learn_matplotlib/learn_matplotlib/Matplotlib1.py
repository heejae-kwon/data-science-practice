import matplotlib.pyplot as plt
import numpy as np

class Matplotlib1:
    def __init__(self):
        return
    
    def draw_graph(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        plt.plot(x, y)
        plt.title("My Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        return
    
    def save_fig1(self):
        x = [1, 2, 3]
        y = [1, 2, 3]
        plt.plot(x, y)
        plt.title("My Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig('picture.png')
    
    def save_fig2(self):
        x = np.linspace(0, np.pi * 10, 500) # PI * 10 너비에, 500개의 점을 균일하게 찍기
        fig, axes = plt.subplots(2, 1) # 2개의 그래프가 들어가는 Figure 생성
        axes[0].plot(x, np.sin(x)) # 첫 번째 그래프는 사인(Sin) 그래프
        axes[1].plot(x, np.cos(x)) # 두 번째 그래프는 코사인(Cos) 그래프
        fig.savefig("sin&cos.png")
    
    def draw_linegraph1(self):
        x = np.arange(-9, 10)
        y = x ** 2
        # 라인 스타일로는 '-', ':', '-.', '--' 등이 사용될 수 있습니다.
        plt.plot(x, y, linestyle=":", marker="*")
        # X축 및 Y축에서 특정 범위를 자를 수도 있습니다.
        plt.show()
    
    def draw_linegraph2(self):
        x = np.arange(-9, 10)
        y1 = x ** 2
        y2 = -x
        plt.plot(x, y1, linestyle="-.", marker="*", color="red", label="y = x * x")
        plt.plot(x, y2, linestyle=":", marker="o", color="blue", label="y = -x")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(
        shadow=True,
        borderpad=1
        )
        plt.show()

    def run(self):
        self.draw_graph()
        self.save_fig1()
        self.save_fig2()
        self.draw_linegraph1()
        self.draw_linegraph2()

        return



    