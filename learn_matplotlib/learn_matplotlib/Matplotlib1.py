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


    def run(self):
        self.draw_graph()


        return



    