# utils/feature_engineering.py

import numpy as np
from gensim.models import Word2Vec
import re

def train_word2vec_model(sentences, vector_size=1000, window=30, min_count=1, sample=1e-3, workers=4):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sample=sample,
        workers=workers
    )
    return model

def get_average_word2vec(tokens_list, model, vector_size):
    averaged_vectors = []
    for tokens in tokens_list:
        valid_vectors = [model.wv[token] for token in tokens if token in model.wv]
        if valid_vectors:
            averaged_vectors.append(np.mean(valid_vectors, axis=0))
        else:
            averaged_vectors.append(np.zeros(vector_size))
    return np.array(averaged_vectors)
