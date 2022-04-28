import numpy as np
from jax import jit, grad, hessian
import jax.numpy as jnp
import scipy.optimize
from functools import partial


class LogisticRegressionOptimized:
    def __init__(self):
        self.w = None
        self.losses = []

    @staticmethod
    def accuracy(y, y_hat):
        accuracy = jnp.sum(y == y_hat) / len(y)
        return accuracy

    def get_weights(self):
        return self.w

    def get_losses(self):
        return self.losses

    @staticmethod
    def loss(w, X, y):
        margin = jnp.dot(X, w)
        l_if_pos = -jnp.logaddexp(0, -margin) * y
        l_if_neg = -jnp.logaddexp(0, margin) * (1 - y)

        l = -(l_if_pos + l_if_neg)

        return jnp.sum(l)

    def train(self, X, y):
        # Initializing weights to zero.
        self.w = np.zeros(X.shape[1])

        norm_X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

        # Find gradient and hessian
        loss_to_jit = partial(self.loss, X=norm_X, y=y)
        jit_loss = jit(loss_to_jit)
        jit_loss_grad = jit(grad(loss_to_jit))
        jit_loss_hess = jit(hessian(loss_to_jit))

        # Train
        self.w = scipy.optimize.fmin_ncg(jit_loss, x0=self.w, fprime=jit_loss_grad, fhess=jit_loss_hess)

    def predict(self, X, threshold=0.5):

        norm_X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)
        # Calculating predictions/y_hat.
        preds = 1.0/(1 + jnp.exp(-(jnp.dot(norm_X, self.w))))
        # Classify as 1 if prediction is above or equal to threshold
        pred_class = (preds >= threshold).astype(int)

        return pred_class
