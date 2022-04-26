import numpy as np


class LogisticRegression:
    def __init__(self, max_epochs, learning_rate, batch_size):
        self.w = None
        self.b = 0
        self.losses = []
        self.epochs = max_epochs
        self.lr = learning_rate
        self.bs = batch_size

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    @staticmethod
    def _gradients(X, y, y_hat):
        m = X.shape[0]

        # Gradient of loss w.r.t weights
        dw = (1 / m) * np.dot(X.T, (y_hat - y))

        # Gradient of loss w.r.t bias
        db = (1 / m) * np.sum((y_hat - y))

        return dw, db

    @staticmethod
    def _normalize(X):
        m, n = X.shape

        for i in range(n):
            X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X

    @staticmethod
    def accuracy(y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy

    def get_weights(self):
        return self.w

    def get_bias(self):
        return self.b

    def get_losses(self):
        return self.losses

    def loss(self, X, y):
        margin = np.dot(X, self.w)
        return y * -np.logaddexp(0, np.exp(margin) + (1 - y) * (1 + np.logaddexp(0, np.exp(margin))))

    def train(self, X, y):
        m, n = X.shape

        # Initializing weights and bias to zeros.
        self.w = np.zeros((n, 1))
        self.b = 0
        self.losses = []

        # Reshape y.
        y = y.reshape(m, 1)

        # Normalize inputs
        x = self._normalize(X)

        # Train
        for epoch in range(self.epochs):
            for i in range((m - 1) // self.bs + 1):
                # Defining batches for SGD (this can be changed)
                start_i = i * self.bs
                end_i = start_i + self.bs
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]

                # Predict
                y_hat = self._sigmoid(np.dot(xb, self.w) + self.b)

                # Calculate gradients
                dw, db = self._gradients(xb, yb, y_hat)

                # Update params
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # Calc loss
            self.losses.append(self.loss(x, y))

    def predict(self, X):
        assert self.w is not None
        # Normalizing the inputs.
        x = self._normalize(X)

        # Calculating presictions/y_hat.
        preds = self._sigmoid(np.dot(X, self.w) + self.b)

        # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]

        return np.array(pred_class)
