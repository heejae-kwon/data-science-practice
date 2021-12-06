import numpy as np
import pandas as pd
from pathlib import Path

class Pandas2:
    def __init__(self):
        return
    
    def run_dataFrame_nullcheck(self):
        word_dict = {
            'Apple': '사과',
            'Banana': '바나나',
            'Carrot': '당근',
            'Durian': '두리안'
        }

        frequency_dict = {
            'Apple': 3,
            'Banana': 5,
            'Carrot': np.nan,
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
        print(summary.notnull())
        print(summary.isnull())
        summary['frequency'] = summary['frequency'].fillna('데이터 없음')
        print(summary)
    

    def run_series_operation(self):
        array1 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
        array2 = pd.Series([4, 5, 6], index=['B', 'C', 'D'])
        array = array1.add(array2, fill_value=0)
        print(array)
    
    def run_dataFrame_operation(self):
        array1 = pd.DataFrame([[1, 2], [3, 4]], index=['A', 'B'])
        array2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['B', 'C', 'D'])

        print(array1)
        print(array2)

        array = array1.add(array2, fill_value=0)
        print(array)

    def run_dataFrame_add(self):
        array1 = pd.DataFrame([[1, 2], [3, 4]], index=['A', 'B'])
        array2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['B', 'C', 'D'])

        array = array1.add(array2, fill_value=0)
        print(array)
        print("컬럼 1의 합:", array[1].sum())
        print(array.sum())

    def run_dataFrame_sort(self):
        word_dict = {
            'Apple': '사과',
            'Banana': '바나나',
            'Carrot': '당근',
            'Durian': '두리안'
        }

        frequency_dict = {
            'Apple': 3,
            'Banana': 5,
            'Carrot': 1,
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
        summary = summary.sort_values('frequency', ascending=False)
        print(summary)
                        

    def run(self):
        self.run_dataFrame_nullcheck()
        self.run_series_operation()
        self.run_dataFrame_operation()
        self.run_dataFrame_add()
        self.run_dataFrame_sort()

        return


    