import torch
import pickle
import numpy as np
import pandas as pd
from generate_vocab import tokenize, generate_glove_weights, generate_vocab_and_glove
from torch.utils.data import Dataset

# Bidirectional GRU with Trainable Embeddings Model
class Net(torch.nn.Module):
    def __init__(self, weights, maxlen):
        super(Net, self).__init__()
        self.hidden_size = 128
        self.feature_size = 300
        self.keep_size = 0.67
        self.maxlen = maxlen
        self.gru = torch.nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size, num_layers=1,
                                bidirectional=True, dropout=self.keep_size)
        self.dense1 = torch.nn.Linear(2 * self.hidden_size, 1)
        self.dense2 = torch.nn.Linear(self.maxlen, 1)
        self.embedding = torch.nn.Embedding.from_pretrained(
            weights, freeze=False)

    def forward(self, x, hidden):
        x = self.embedding(x)
        gru_out, next_hidden = self.gru(x, hidden)
        logits = self.dense1(gru_out) + 1e-8
        logits = torch.squeeze(logits)
        logits = self.dense2(logits)
        return torch.sigmoid(logits), next_hidden


def train(model, dataset, max_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    model.train()
    for epoch in range(10):
        print("epoch", epoch)
        hidden = torch.zeros(2, max_len, model.hidden_size)
        count = 0
        for inputs, labels in dataset:
            model.zero_grad()
            optimizer.zero_grad()
            outputs, _ = model(inputs, hidden)
            outputs = outputs[:, -1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 10 == 0:
                print("count:", count)


def test(model, dataset, max_len):
    hidden = torch.zeros(2, max_len, model.hidden_size)
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in dataset:
        outputs, _ = model(inputs, hidden)
        labels = (torch.eq(labels, 1.0)) * 1
        outputs = (torch.gt(outputs, 0.5)) * 1
        outputs = torch.squeeze(outputs)
        total += len(inputs)
        correct += float((labels == outputs).sum())
    return correct / total


def prep_data_from_csv(csv_file, vocab):
    data = pd.read_csv(csv_file)
    data['Party'] = data['Party'].apply(lambda x: 1 if x == 'Democrat' else 0)
    data['Tweet'] = data['Tweet'].apply(tokenize)
    max_len = data['Tweet'].apply(len).max()
    data['Tweet'] = data['Tweet'].apply(lambda x: [vocab[token] if token in vocab else "<UNK>" for token in x])
    inputs = data['Tweet'].tolist()
    labels = data['Party'].tolist()
    return inputs, labels, max_len


if __name__ == "__main__":
    # Load vocab and GLoVe embeddings
    pkl_file = open('data/vocab.pkl', 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    glove = torch.Tensor(np.load("data/glove.npy", allow_pickle=True))

    # Prepare data from CSV
    inputs, labels, max_len = prep_data_from_csv('data/ExtractedTweets.csv', vocab)

    # Convert inputs and labels to tensors and create DataLoader
    inputs_tensor = torch.LongTensor(inputs)
    labels_tensor = torch.Tensor(labels)
    data = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
    dataset = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)

    # Initialize Model
    model = Net(glove, max_len)

    # Pre-test to compare accuracy later
    accuracy = test(model, dataset, max_len)
    print("Accuracy 1:", accuracy)

    # Train model
    train(model, dataset, max_len)

    # Save weights if desired
    torch.save(model, "models/model80acc.pt")

    # Test model and print accuracy
    accuracy = test(model, dataset, max_len)
    print("Accuracy 2:", accuracy)