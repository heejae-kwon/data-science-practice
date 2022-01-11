import torch
#from glove import Corpus, Glove
import torch.nn as nn
# https://wikidocs.net/64779


class Embedding:
    def __init__(self):

        return

    def __glove(self):
        #corpus = Corpus()
        # Word2Vec 전처리 result
        corpus.fit(result, window=5)
        # 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성

        glove = Glove(no_components=100, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        # 학습에 이용할 쓰레드의 개수는 4로 설정, 에포크는 20.

        return

    def __without_nn_Embedding(self):
        train_data = 'you need to know how to code'

        # 중복을 제거한 단어들의 집합인 단어 집합 생성.
        word_set = set(train_data.split())

        # 단어 집합의 각 단어에 고유한 정수 맵핑.
        vocab = {word: i+2 for i, word in enumerate(word_set)}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
        print(vocab)

        # 단어 집합의 크기만큼의 행을 가지는 테이블 생성.
        embedding_table = torch.FloatTensor([
            [0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0],
            [0.2,  0.9,  0.3],
            [0.1,  0.5,  0.7],
            [0.2,  0.1,  0.8],
            [0.4,  0.1,  0.1],
            [0.1,  0.8,  0.9],
            [0.6,  0.1,  0.1]])

        sample = 'you need to run'.split()
        idxes = []

        # 각 단어를 정수로 변환
        for word in sample:
            try:
                idxes.append(vocab[word])
        # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
            except KeyError:
                idxes.append(vocab['<unk>'])
        idxes = torch.LongTensor(idxes)

        # 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
        lookup_result = embedding_table[idxes, :]
        print(lookup_result)

    def __with_nn_Embedding(self):
        train_data = 'you need to know how to code'

        # 중복을 제거한 단어들의 집합인 단어 집합 생성.
        word_set = set(train_data.split())

        # 단어 집합의 각 단어에 고유한 정수 맵핑.
        vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
        embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                                       embedding_dim=3,
                                       padding_idx=1)

        print(embedding_layer.weight)
    
    def __

    def run(self):
        self.__without_nn_Embedding()
        self.__with_nn_Embedding()

        return
