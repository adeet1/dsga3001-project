import numpy as np


class LogisticRegressionOriginal:
    def __init__(self, max_epochs, learning_rate, batch_size):
        self.w = None
        self.b = 0
        self.losses = []
        self.epochs = max_epochs
        self.lr = learning_rate
        self.bs = batch_size

    @staticmethod
    def _array_sum(A):
        total = 0
        for a in A.flat:
            total += a
        return total

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1 + np.exp(-z))

    def array_mean(self, A, axis=None):
        if axis is None:
            return self._array_sum(A) / len(A.flat)
        elif axis == 0:
            output = np.zeros(A.shape[1])
            for c in range(A.shape[1]):
                output[c] = self._array_sum(A[:, c]) / A.shape[0]
            return output
        elif axis == 1:
            output = np.zeros(A.shape[0])
            for r in range(A.shape[0]):
                output[r] = self._array_sum(A[r, :]) / A.shape[1]
            return output

    def array_std(self, A, axis=None):
        if axis is None:
            ex = self.array_mean(A)
            ex2 = self.array_mean(A ** 2)
            return np.sqrt(ex2 - (ex ** 2))
        elif axis == 0:
            output = np.zeros(A.shape[1])
            for c in range(A.shape[1]):
                ex = self.array_mean(A[:, c])
                ex2 = self.array_mean(A[:, c] ** 2)
                output[c] = ex2 - (ex ** 2)
            return np.sqrt(output)
        elif axis == 1:
            output = np.zeros(A.shape[0])
            for r in range(A.shape[0]):
                ex = self.array_mean(A[r, :])
                ex2 = self.array_mean(A[r, :] ** 2)
                output[r] = ex2 - (ex ** 2)
            return np.sqrt(output)

    def dot_product(self, A, B):
        assert A.shape[1] == B.shape[0], f'{A.shape} and {B.shape} incompatible for dot product'
        output = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                output[i, j] = self._array_sum(A[i, :] * B[:, j])
        return output

    def gradients(self, X, y, y_hat):
        m = X.shape[0]

        # Gradient of loss w.r.t weights
        dw = (1 / m) * self.dot_product(X.T, (y_hat - y))

        # Gradient of loss w.r.t bias
        db = (1 / m) * self._array_sum((y_hat - y))

        return dw, db

    def normalize(self, X):
        X = (X - self.array_mean(X, axis=0)) / self.array_std(X, axis=0)

        return X

    def get_weights(self):
        return self.w

    def get_bias(self):
        return self.b

    def get_losses(self):
        return self.losses

    def loss(self, X, y):
        margin = self.dot_product(X, self.w)
        l_if_pos = -np.logaddexp(0, -margin) * y
        l_if_neg = -np.logaddexp(0, margin) * (1 - y)

        l = -(l_if_pos + l_if_neg)

        return self._array_sum(l)

    def train(self, X, y):

        m, n = X.shape

        # Initializing weights and bias to zeros.
        self.w = np.zeros((n, 1))
        self.b = 0

        # Reshape y.
        y = y.reshape(m, 1)

        # Normalize inputs
        x = self.normalize(X)

        # Store losses
        self.losses = []

        # Train
        for epoch in range(self.epochs):
            for i in range((m - 1) // self.bs + 1):
                # Defining batches for SGD (this can be changed)
                start_i = i * self.bs
                end_i = start_i + self.bs
                xb = x[start_i:end_i]
                yb = y[start_i:end_i]

                # Predict
                y_hat = self._sigmoid(self.dot_product(xb, self.w) + self.b)

                # Calculate gradients
                dw, db = self.gradients(xb, yb, y_hat)

                # Update params
                self.w -= self.lr * dw
                self.b -= self.lr * db

            # Calc loss
            l = self.loss(X, y)
            self.losses.append(l)

    def predict(self, X):
        assert self.w is not None
        # Normalizing the inputs.
        X = self.normalize(X)

        # Calculating presictions/y_hat.
        preds = self._sigmoid(self.dot_product(X, self.w) + self.b)

        # if y_hat >= 0.5 --> round up to 1
        # if y_hat < 0.5 --> round up to 1
        pred_class = [1 if i > 0.5 else 0 for i in preds]

        return np.array(pred_class)

    def accuracy(self, y: np.ndarray, y_hat: np.ndarray):
        accuracy = self._array_sum((y == y_hat)) / len(y)
        return accuracy
