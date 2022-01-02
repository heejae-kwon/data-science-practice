import spacy
import nltk
import urllib.request
import pandas as pd
from nltk.tokenize import word_tokenize
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
from torchtext.legacy import data  # torchtext.data 임포트

# https://wikidocs.net/64517


class NLP:
    def __init__(self):
        nltk.download('punkt')
        self.__spacy_en = spacy.load('en_core_web_sm')
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

        return

    def __tokenize(self, en_text):
        return [tok.text for tok in self.__spacy_en.tokenizer(en_text)]

    def __tokenization(self):
        en_text = "A Dog Run back corner near spare bedrooms"
        # spaCy
        print(self.__tokenize(en_text))
        # nltk
        print(word_tokenize(en_text))
        # 띄어쓰기
        print(en_text.split())

        return

    def __torchtext_tutorial(self):
        df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
        df.head()
        print('전체 샘플의 개수 : {}'.format(len(df)))

        train_df = df[:25000]
        test_df = df[25000:]
        train_df.to_csv("train_data.csv", index=False)
        test_df.to_csv("test_data.csv", index=False)

        # 필드 정의
        TEXT = data.Field(sequential=True,
                          use_vocab=True,
                          tokenize=str.split,
                          lower=True,
                          batch_first=True,
                          fix_length=20)

        LABEL = data.Field(sequential=False,
                           use_vocab=False,
                           batch_first=False,
                           is_target=True)

        train_data, test_data = TabularDataset.splits(
            path='.', train='train_data.csv', test='test_data.csv', format='csv',
            fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

        print('훈련 샘플의 개수 : {}'.format(len(train_data)))
        print('테스트 샘플의 개수 : {}'.format(len(test_data)))
        print(vars(train_data[0]))
        # 필드 구성 확인.
        print(train_data.fields.items())
        TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
        print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
        print(TEXT.vocab.stoi)

        batch_size = 5
        train_loader = Iterator(dataset=train_data, batch_size=batch_size)
        test_loader = Iterator(dataset=test_data, batch_size=batch_size)
        print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
        print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))
        batch = next(iter(train_loader))  # 첫번째 미니배치
        print(type(batch))
        print(batch.text)

        batch = next(iter(train_loader))  # 첫번째 미니배치
        print(batch.text[0])  # 첫번째 미니배치 중 첫번째 샘플

        return

    def __torchtext_batch_first(self):
        df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
        # 전체 샘플의 개수를 보겠습니다.
        print('전체 샘플의 개수 : {}'.format(len(df)))
        train_df = df[:25000]
        test_df = df[25000:]
        train_df.to_csv("train_data.csv", index=False)
        test_df.to_csv("test_data.csv", index=False)

        batch_first = False
        # 필드 정의
        TEXT = data.Field(sequential=True,
                          use_vocab=True,
                          tokenize=str.split,
                          lower=True,
                          batch_first=batch_first,  # <== 이 부분을 True로 합니다.
                          fix_length=20)

        LABEL = data.Field(sequential=False,
                           use_vocab=False,
                           batch_first=False,
                           is_target=True)

        # TabularDataset은 데이터를 불러오면서 필드에서 정의했던 토큰화 방법으로 토큰화를 수행합니다.
        train_data, test_data = TabularDataset.splits(
            path='.', train='train_data.csv', test='test_data.csv', format='csv',
            fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

        # 정의한 필드에 .build_vocab() 도구를 사용하면 단어 집합을 생성합니다.
        # 10,000개의 단어를 가진 단어 집합 생성
        TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

        # 배치 크기를 정하고 첫번째 배치를 출력해보겠습니다.
        batch_size = 5
        train_loader = Iterator(dataset=train_data, batch_size=batch_size)
        batch = next(iter(train_loader))  # 첫번째 미니배치
        print(batch.text)
        print(batch.text.shape)
        return

    def run(self):
        self.__tokenization()
        self.__torchtext_tutorial()
        self.__torchtext_batch_first()

        return
