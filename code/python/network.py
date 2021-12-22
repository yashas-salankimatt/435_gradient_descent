import numpy as np
import pandas as pd

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegression(object):
    def __init__(self):
        # W^TX: (14x14)x(14x1) [Hidden size: 1]
        self.w1 = np.random.rand(14, 1) - .5

        self.lr = 0.01


    def _gradient(self, x, y, z):
        return x @ (z - y)

    def bce_loss(self, y, z):
        return -(y * np.log(z+1e-30) + (1-y) * np.log(1-z+1e-30))

    def forward(self, x):
        z = self.w1.T @ x
        out = sigmoid(z)
        return out

    def backward(self, x, y, z):
        self.w1 = self.w1 - self.lr * self._gradient(x, y, z)

    def train(self):
        dataset = pd.read_csv("dataset.csv", index_col=0)
        features = dataset.drop(columns='14').values
        labels = dataset['14'].values

        running_loss = 0
        for epoch in range(50):
            for i, (feature, label) in enumerate(zip(features, labels)):
                output = self.forward(feature.reshape(-1, 1))
                loss = self.bce_loss(label, output)
                running_loss += loss
                self.backward(feature.reshape(-1, 1), label, output)
            print("Loss:", running_loss)
            if epoch == 5:
                print(self.w1)
                break
            running_loss = 0


logreg = LogisticRegression()
logreg.train()
