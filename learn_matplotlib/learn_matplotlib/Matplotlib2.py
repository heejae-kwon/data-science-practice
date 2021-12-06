import matplotlib.pyplot as plt
import numpy as np

class Matplotlib2:
    def __init__(self):
        return

    def draw_linegraph(self):
        x = np.arange(-9, 10)
        y1 = x ** 2
        plt.plot(
            x, y1,
            linestyle=":",
            marker="o",
            markersize=8,
            markerfacecolor="blue",
            markeredgecolor="red"
        )
        plt.show()
    
    def draw_bargraph(self):
        x = np.arange(-9, 10)
        plt.bar(x, x ** 2)
        plt.show()
    
    def draw_stackbargraph(self):
        x = np.random.rand(10) # 아래 막대
        y = np.random.rand(10) # 중간 막대
        z = np.random.rand(10) # 위 막대
        data = [x, y, z]
        x_array = np.arange(10)
        for i in range(0, 3): # 누적 막대의 종류가 3개
            plt.bar(
                x_array, # 0부터 10까지의 X 위치에서
                data[i], # 각 높이(10개)만큼 쌓음
                bottom=np.sum(data[:i], axis=0)
            )
        plt.show()
    
    def draw_scattergraph(self):
        x = np.random.rand(10)
        y = np.random.rand(10)
        colors = np.random.randint(0, 100, 10)
        sizes = np.pi * 1000 * np.random.rand(10)
        plt.scatter(x, y, c=colors, s=sizes, alpha=0.7)
        plt.show()
                                    

    def run(self):
        self.draw_linegraph()
        self.draw_bargraph()
        self.draw_stackbargraph()
        self.draw_scattergraph()

        return



    