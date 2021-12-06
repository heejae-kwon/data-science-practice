import numpy as np
import pandas as pd
from pathlib import Path

class Pandas1:
    def __init__(self):
        return

    def run_series1(self):
        array = pd.Series(['사과', '바나나', '당근'], index=['a', 'b', 'c'])
        print(array)
        print(array['a'])
    
    def run_series2(self):
        data = {
           'a': '사과',
           'b': '바나나',
           'c': '당근'
        }
    # Dict 자료형을 Series로 바꾸기
        array = pd.Series(data)
        print(array['a'])
    
    def run_dataFrame(self):
        word_dict = {
    'Apple': '사과',
    'Banana': '바나나',
    'Carrot': '당근'
        }

        frequency_dict = {
    'Apple': 3,
    'Banana': 5,
    'Carrot': 7
        }

        word = pd.Series(word_dict)
        frequency = pd.Series(frequency_dict)

        # 이름(Name): 값(Values)
        summary = pd.DataFrame({
           'word': word,
           'frequency': frequency
        })
        print(summary)

    def run_series_calculation(self):
        word_dict = {
    'Apple': '사과',
    'Banana': '바나나',
    'Carrot': '당근'
        }

        frequency_dict = {
    'Apple': 3,
    'Banana': 5,
    'Carrot': 7
        }

        importance_dict = {
    'Apple': 3,
    'Banana': 2,
    'Carrot': 1
        }

        word = pd.Series(word_dict)
        frequency = pd.Series(frequency_dict)
        importance = pd.Series(importance_dict)

        summary = pd.DataFrame({
    'word': word,
    'frequency': frequency,
    'importance': importance
        })

        score = summary['frequency'] * summary['importance']
        summary['score'] = score
        print(summary)
    
    def run_dataFrame_slicing(self):
        word_dict = {
    'Apple': '사과',
    'Banana': '바나나',
    'Carrot': '당근',
    'Durian': '두리안'
}

        frequency_dict = {
    'Apple': 3,
    'Banana': 5,
    'Carrot': 7,
    'Durian': 2
}

        importance_dict = {
    'Apple': 3,
    'Banana': 2,
    'Carrot': 1,
    'Durian': 1
}

        word = pd.Series(word_dict)
        frequency = pd.Series(frequency_dict)
        importance = pd.Series(importance_dict)

        summary = pd.DataFrame({
    'word': word,
    'frequency': frequency,
    'importance': importance
})

        print(summary)

        # 이름을 기준으로 슬라이싱
        print(summary.loc['Banana':'Carrot', 'importance':])

        # 인덱스를 기준으로 슬라이싱
        print(summary.iloc[1:3, 2:])
    
    def run_dataFrame_calculation(self):
        word_dict = {
    'Apple': '사과',
    'Banana': '바나나',
    'Carrot': '당근',
    'Durian': '두리안'
}

        frequency_dict = {
            'Apple': 3,
            'Banana': 5,
            'Carrot': 7,
            'Durian': 2
        }

        importance_dict = {
            'Apple': 3,
            'Banana': 2,
            'Carrot': 1,
            'Durian': 1
        }

        word = pd.Series(word_dict)
        frequency = pd.Series(frequency_dict)
        importance = pd.Series(importance_dict)

        summary = pd.DataFrame({
            'word': word,
            'frequency': frequency,
            'importance': importance
        })

        print(summary)

        summary.loc['Apple', 'importance'] = 5 # 데이터의 변경
        summary.loc['Elderberry'] = ['엘더베리', 5, 3] # 새 데이터 삽입

        print(summary)
    
    def run_csv(self):
        word_dict = {
            'Apple': '사과',
            'Banana': '바나나',
            'Carrot': '당근'
        }

        frequency_dict = {
            'Apple': 3,
            'Banana': 5,
            'Carrot': 7
        }

        word = pd.Series(word_dict)
        frequency = pd.Series(frequency_dict)

        summary = pd.DataFrame({
            'word': word,
            'frequency': frequency
        })

        summary.to_csv("summary.csv", encoding="utf-8-sig")
        saved = pd.read_csv("summary.csv", index_col=0)
        print(saved)
        

    def run(self):
        self.run_series1()
        self.run_series2()
        self.run_dataFrame()
        self.run_series_calculation()
        self.run_dataFrame_slicing()
        self.run_dataFrame_calculation()
        self.run_csv()


        return


    