import numpy as np
import pandas as pd

class Pandas3:
    def __init__(self):
        return
    
    def run_dataFrame_masking(self):
        df = pd.DataFrame(np.random.randint(1, 10, (2, 2)), index=[0, 1], columns=["A", "B"])
        # 데이터 프레임 출력하기
        print(df)
        # 컬럼 A의 각 원소가 5보다 작거나 같은지 출력
        print(df["A"] <= 5)
        # 컬럼 A의 원소가 5보다 작고, 컬럼 B의 원소가 8보다 작은 행 추출
        print(df.query("A <= 5 and B <= 8"))
    
    def run_dataFrame_apply(self):
        df = pd.DataFrame([[1, 2, 3, 4], [1, 2, 3, 4]], index=[0, 1], columns=["A", "B", "C", "D"])
        print(df)

        df = df.apply(lambda x: x + 1)
        print(df)

        def addOne(x):
            return x + 1
        
        df = df.apply(addOne)
        print(df)
    
    def run_dataFrame_replace(self):
        df = pd.DataFrame([
        ['Apple', 'Apple', 'Carrot', 'Banana'],
        ['Durian', 'Banana', 'Apple', 'Carrot']],
        index=[0, 1],
        columns=["A", "B", "C", "D"])

        print(df)
        df = df.replace({"Apple": "Airport"})
        print(df)

    def run_dataFrame_groupby_1(self):
        df = pd.DataFrame([
        ['Apple', 7, 'Fruit'],
        ['Banana', 3, 'Fruit'],
        ['Beef', 5, 'Meal'],
        ['Kimchi', 4, 'Meal']],
        columns=["Name", "Frequency", "Type"])

        print(df)
        print(df.groupby(['Type']).sum())
    
    def run_dataFrame_groupby_2(self):
        df = pd.DataFrame([
        ['Apple', 7, 5, 'Fruit'],
        ['Banana', 3, 6, 'Fruit'],
        ['Beef', 5, 2, 'Meal'],
        ['Kimchi', 4, 8, 'Meal']],
        columns=["Name", "Frequency", "Importance", "Type"])

        print(df)
        print(df.groupby(["Type"]).aggregate([min, max, np.average]))
    
    def run_dataFrame_groupby_3(self):
        df = pd.DataFrame([
        ['Apple', 7, 5, 'Fruit'],
        ['Banana', 3, 6, 'Fruit'],
        ['Beef', 5, 2, 'Meal'],
        ['Kimchi', 4, 8, 'Meal']],
        columns=["Name", "Frequency", "Importance", "Type"])

        def my_filter(data):
            return data["Frequency"].mean() >= 5

        print(df)
        df = df.groupby("Type").filter(my_filter)
        print(df)
        return

    def run_dataFrame_groupby_4(self):
        df = pd.DataFrame([
        ['Apple', 7, 5, 'Fruit'],
        ['Banana', 3, 6, 'Fruit'],
        ['Beef', 5, 2, 'Meal'],
        ['Kimchi', 4, 8, 'Meal']],
        columns=["Name", "Frequency", "Importance", "Type"])

        df = df.groupby("Type").get_group("Fruit")
        print(df)
        return

    def run_dataFrame_groupby_5(self):
        df = pd.DataFrame([
        ['Apple', 7, 5, 'Fruit'],
        ['Banana', 3, 6, 'Fruit'],
        ['Beef', 5, 2, 'Meal'],
        ['Kimchi', 4, 8, 'Meal']],
        columns=["Name", "Frequency", "Importance", "Type"])

        df["Gap"] = df.groupby("Type")["Frequency"].apply(lambda x: x - x.mean())
        print(df)
        return
                                                                
    def run_dataFrame_multiplexing(self):
        df = pd.DataFrame(
        np.random.randint(1, 10, (4, 4)),
        index=[['1차', '1차', '2차', '2차'], ['공격', '수비', '공격', '수비']],
        columns=['1회', '2회', '3회', '4회']
        )
        print(df)
        print(df[["1회", "2회"]].loc["2차"])
    
    def run_dataFrame_pivot_table(self):
        df = pd.DataFrame([
            ['Apple', 7, 5, 'Fruit'],
            ['Banana', 3, 6, 'Fruit'],
            ['Coconut', 2, 6, 'Fruit'],
            ['Rice', 8, 2, 'Meal'],
            ['Beef', 5, 2, 'Meal'],
            ['Kimchi', 4, 8, 'Meal']],
        columns=["Name", "Frequency", "Importance", "Type"])

        print(df)
        df = df.pivot_table(
            index="Importance", columns="Type", values="Frequency",
            aggfunc=np.max
        )
        print(df)


    def run(self):
        self.run_dataFrame_masking()
        self.run_dataFrame_apply()
        self.run_dataFrame_replace()
        self.run_dataFrame_groupby_1()
        self.run_dataFrame_groupby_2()
        self.run_dataFrame_groupby_3()
        self.run_dataFrame_groupby_4()
        self.run_dataFrame_groupby_5()
        self.run_dataFrame_multiplexing()
        self.run_dataFrame_pivot_table()

        return


    