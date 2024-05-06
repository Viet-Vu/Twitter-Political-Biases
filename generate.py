import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# code to generate vocab & glove weights
import string

# Tokenizes a tweet
def tokenize(text):
    words = text.split()
    # For each word, replace with token if its a link or @
    for i in range(len(words)):
        w = words[i]
        if "http" in w:
            words[i] = "<LINK>"
        elif w.startswith("@"):
            words[i] = "<MENTION>"
        elif w.startswith(".@"):
            words[i] = "<MENTION>"
        else:
            # Remove punctuation
            words[i] = w.strip().strip(string.punctuation).strip(
                '…’‘”“ ,.•——?"\'').lower()
    return words

# Generate and save GLoVe embeddings for our vocabulary
def generate_glove_weights(vocab):
    embeddings_index = {}
    with open("data/glove.42B.300d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocab), 300))
    hits = 0.0
    for i, word in enumerate(vocab):
        if word.lower() in embeddings_index:
            hits += 1
            embedding_matrix[i] = embeddings_index[word.lower()]
        else:
            # If word does not have pretrained GLoVe embedding, randomly initialize an embedding for it
            embedding_matrix[i] = np.random.randn(1, 300)
    print("Word hit rate %f" % (hits/len(vocab)))
    return embedding_matrix

def generate_vocab_and_glove():
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<LINK>": 2,
        "<MENTION>": 3
    }

    data=pd.read_csv("data/ExtractedTweets.csv")

    # Add tweets to vocab
    for tweet in data.Tweet:
        tweet = tokenize(tweet)
        for word in tweet:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    words_to_unk = set()

    # Remove words that appear fewer than 6 times
    index = 0
    for word in vocab:
        if vocab[word] > 5:
            vocab[word] = index
            index += 1
        else:
            words_to_unk.add(word)

    for word in words_to_unk:
        del vocab[word]

    with open("data/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    glove = generate_glove_weights(vocab)
    np.save("data/glove.npy", glove)

    return vocab, glove

generate_vocab_and_glove()
