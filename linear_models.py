import numpy as np
from typing import Dict, Tuple, Iterable, Optional, Any

"""
Linear Models skeleton for Deep Learning HW1.

This module defines the classes and functions that students need to complete.
The notebook will import these definitions instead of defining them inline.

Each function or method marked with `raise NotImplementedError` needs to be
implemented by the student.
"""


class LinearClassifier:
    """
    Base class for linear classifiers.  Stores weights and provides the
    interface for prediction, training, and computing accuracy.  Subclasses
    should override ``predict`` and ``loss`` to implement specific
    classification algorithms.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the classifier with a small random weight matrix.

        Parameters
        ----------
        X : np.ndarray
            Training data matrix of shape (N, D), where N is the number
            of samples and D is the number of features (possibly including
            a bias term).  The weight matrix will have shape (D, C).

        y : np.ndarray
            Array of shape (N,) containing integer class labels in the
            range [0, C-1], where C is the number of distinct classes.
        """
        N, D = X.shape
        C = int(np.max(y)) + 1
        # Initialize weights with small random values
        self.W = 0.001 * np.random.randn(D, C)
        self.num_classes = C
        self.num_features = D

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given data using the classifier's weights.

        This default implementation should be overridden by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (M, D) of input data.

        Returns
        -------
        np.ndarray
            Array of shape (M,) of predicted class labels.
        """
        raise NotImplementedError("predict method must be implemented in subclass")

    def calc_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy on a dataset.

        Accuracy is the fraction of instances that are classified correctly.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        y : np.ndarray
            True labels of shape (M,).

        Returns
        -------
        float
            Accuracy as a float in the range [0, 1].
        """
        accuracy = 0.0
        ###########################################################################
        # TODO: Implement this method.                                            #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################

        # predict labels for every sample in X and compare to the ground truth.
        # accuracy is the fraction of samples where the prediction equals the true label.
        y_pred = self.predict(X)
        accuracy = float(np.mean(y_pred == y))

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return accuracy

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              learning_rate: float = 1e-3,
              num_iters: int = 100,
              batch_size: int = 200,
              verbose: bool = False) -> list:
        """
        Train the classifier using stochastic gradient descent.

        This method samples minibatches, computes the loss and gradient,
        and performs weight updates.  It collects the loss value at each
        iteration in ``loss_history`` and returns it.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        learning_rate : float, optional
            Step size for gradient descent (default: 1e-3).

        num_iters : int, optional
            Number of iterations to run (default: 100).

        batch_size : int, optional
            Number of samples per minibatch (default: 200).

        verbose : bool, optional
            If True, prints loss every 100 iterations.

        Returns
        -------
        list
            A list containing the loss value at each iteration.
        """
        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in every iteration.                       #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch                                                               #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #                                                                       #
        # Next, calculate the loss and gradient and update the weights using    #
        # the learning rate. Use the loss_history array to save the loss on     #
        # iteration to visualize the loss.                                      #
        #########################################################################
        num_instances = X.shape[0]
        loss_history = []
        loss = 0.0
        for i in range(num_iters):
            X_batch = None
            y_batch = None
            ###########################################################################
            # TODO: Create X_batch and y_batch. Call the loss method to get the loss value  #
            # and grad (the loss function is being override, see the loss             #
            # function return values).                                                #
            # Finally, append each of the loss values created in each iteration       #
            # to loss_history.                                                        #
            ###########################################################################
            #                          START OF YOUR CODE                             #
            ###########################################################################

            # sample a mini-batch of size `batch_size` with replacement.
            # sampling with replacement is faster and is the standard sgd choice here.
            batch_idx = np.random.choice(num_instances, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            # compute loss and gradient for the current batch.
            # note: self.loss is overridden by each subclass to pick the correct loss function.
            loss, grad = self.loss(X_batch, y_batch)
            # keep the loss for later plotting of the training curve.
            loss_history.append(loss)

            ###########################################################################
            #                           END OF YOUR CODE                              #
            ###########################################################################
            # TODO:                                                                   #
            # Perform parameter update                                                #
            # Update the weights using the gradient and the learning rate.            #
            ###########################################################################
            #                          START OF YOUR CODE                             #                                                         #
            ###########################################################################

            # basic vanilla gradient descent step: move weights against the gradient.
            self.W -= learning_rate * grad

            ###########################################################################
            #                       END OF YOUR CODE                                  #
            ###########################################################################

            if verbose and i % 100 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def loss(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the loss function and its gradient.

        Subclasses should override this method to compute the appropriate
        loss and gradient for their algorithm.

        Parameters
        ----------
        X : np.ndarray
            Minibatch of data of shape (N, D).

        y : np.ndarray
            Labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` is
            an array of the same shape as ``self.W``.
        """
        raise NotImplementedError("loss must be implemented in subclass")


class LinearPerceptron(LinearClassifier):
    """
    Linear classifier that uses the Perceptron loss.
    Students should implement the ``predict`` and ``loss`` methods.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the perceptron using the base class constructor.
        """
        ###########################################################################
        # TODO: Initiate the parameters of your model.                            #
        # You can assume y takes values 0...K-1 where K is number of classes      #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################

        # delegate to the base class, which creates a small random (D, C) weight matrix
        # and stores num_classes / num_features on self.
        super().__init__(X, y)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the perceptron rule.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        y_pred = None
        ###########################################################################
        # TODO: Implement this method.                                            #
        ###########################################################################
        #                          START OF YOUR CODE                             #
        ###########################################################################

        # compute raw class scores and pick the class with the highest score per sample.
        scores = X @ self.W             # shape (M, C)
        y_pred = np.argmax(scores, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute perceptron loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        return perceptron_loss_naive(self.W, X_batch, y_batch)


class LogisticRegression(LinearClassifier):
    """
    Linear classifier that uses softmax and cross-entropy loss for multiclass
    classification.
    Students should implement the ``predict`` and ``loss`` methods.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the logistic regression model using the base class constructor.
        """
        self.W = None
        ###########################################################################
        # TODO: Initialize the model via the base class constructor.              #
        ###########################################################################

        # use the base class initializer to set up a small random (D, C) weight matrix
        # and record num_classes / num_features for later use.
        super().__init__(X, y)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels using the softmax probabilities.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (M, D).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (M,).
        """
        y_pred = None
        ###########################################################################
        # TODO: Implement this method.                                                  #
        ###########################################################################

        # compute class scores, convert to probabilities, pick the argmax class.
        # note: argmax of softmax equals argmax of raw scores (softmax is monotonic),
        # but we call softmax here to match the section title in the notebook.
        scores = X @ self.W              # shape (M, C)
        probs = softmax(scores)          # shape (M, C), each row sums to 1
        y_pred = np.argmax(probs, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Compute softmax cross-entropy loss and gradient for a minibatch.

        Parameters
        ----------
        X_batch : np.ndarray
            Minibatch data of shape (N, D).

        y_batch : np.ndarray
            Minibatch labels of shape (N,).

        Returns
        -------
        tuple
            A tuple (loss, dW) where ``loss`` is a scalar and ``dW`` has
            the same shape as ``self.W``.
        """
        # will be implemented later
        return softmax_cross_entropy_vectorized(self.W, X_batch, y_batch)

def perceptron_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass perceptron loss using explicit loops.

    This function computes the average multiclass perceptron margin loss over
    the batch and the gradient of the loss with respect to the weight matrix W.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
        Average multiclass perceptron margin loss
    dW : (D, C)
        Gradient of the loss w.r.t. W
    """
    N, D = X.shape
    _, C = W.shape

    # Initialize loss & gradient
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Implement Perceptron loss with explicit loops                     #
    #                                                                         #
    # After looping over all samples:                                         #
    #   - Average loss and gradient by N                                      #
    #############################################################################

    # multiclass perceptron margin loss with explicit python loops.
    # for a misclassified sample, the loss is the margin by which the predicted
    # class score beats the true-class score; correctly classified samples
    # contribute zero.
    for i in range(N):
        # compute raw scores for sample i: one score per class.
        scores_i = X[i] @ W               # shape (C,)
        # predicted class is the one with the highest score.
        pred = int(np.argmax(scores_i))
        true_label = int(y[i])
        # only misclassified samples contribute to the loss and the gradient.
        if pred != true_label:
            loss += scores_i[pred] - scores_i[true_label]
            # negative gradient for the correct class: W -= lr*dW will add x_i there,
            # which raises the correct class score next time.
            dW[:, true_label] -= X[i]
            # positive gradient for the wrong class: W -= lr*dW will subtract x_i there,
            # which lowers the wrong class score next time.
            dW[:, pred] += X[i]

    # average over the batch so the loss and gradient scale are independent of
    # batch size.
    loss /= N
    dW /= N


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW

def softmax_cross_entropy(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    N = X.shape[0]
    loss, dW = 0.0, np.zeros_like(W)

    #############################################################################
    # TODO: Implement the forward pass.                                         #       #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################

    # forward pass, sample by sample (non-vectorized reference version).
    # we compute softmax probabilities row by row so the logic is easy to read;
    # the vectorized implementation later does the exact same math in bulk.
    C = W.shape[1]
    probs_all = np.zeros((N, C))
    for i in range(N):
        # raw scores for sample i, one per class.
        scores_i = X[i] @ W                          # shape (C,)
        # subtract the max score for numerical stability (log-sum-exp trick).
        scores_i -= np.max(scores_i)
        # exponentiate and normalize to get a proper probability distribution.
        exp_scores = np.exp(scores_i)
        probs_all[i] = exp_scores / np.sum(exp_scores)


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


    #############################################################################
    # TODO: Compute the loss.                                                   #
    # Use the average negative log-likelihood of the correct class.             #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################

    # pick out the predicted probability of the true class for every sample,
    # then take -log of it. a tiny epsilon guards against log(0) in edge cases.
    correct_probs = probs_all[np.arange(N), y]
    loss = float(np.mean(-np.log(correct_probs + 1e-12)))

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################


    #############################################################################
    # TODO: Backward pass: compute gradient dW.                                 #                           #
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################

    # gradient of softmax cross-entropy wrt scores is (probs - one_hot_labels).
    # subtract 1 from the probability of the correct class for every row.
    dscores = probs_all.copy()
    dscores[np.arange(N), y] -= 1.0
    # chain rule: dL/dW = X^T @ dL/dscores, then average over the batch.
    dW = X.T @ dscores / N

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return loss, dW



def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax function for each row of the input x.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (N, C) where N is the number of samples and C
        is the number of classes.

    Returns
    -------
    probs : (N, C)
        Row-wise probabilities that sum to 1
    """
    probs = np.zeros_like(x)
    #############################################################################
    #                           START OF YOUR CODE                              #
    #############################################################################

    # numerically stable softmax: subtract the per-row max before exp to avoid
    # overflow. this does not change the resulting probabilities.
    shifted = x - np.max(x, axis=1, keepdims=True)
    # exponentiate the shifted scores.
    exp_shifted = np.exp(shifted)
    # normalize each row so the probabilities sum to exactly 1.
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return probs

def softmax_cross_entropy_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Compute the multiclass softmax cross-entropy loss and its gradient
    using a fully vectorized implementation.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (D, C).

    X : np.ndarray
        Data matrix of shape (N, D).

    y : np.ndarray
        Labels of shape (N,).

    Returns
    -------
    loss : float
    dW   : (D, C) gradient of loss wrt W
    """
    N = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Implement the forward pass in a fully vectorized way.               #
    # 1. Compute the scores (N, C) for all samples in X.                        #
    # 2. Stabilize the scores by subtracting the max score in each row.         #
    # 3. Compute the softmax probabilities (N, C) for all samples.              #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################

    # one big matmul replaces the per-sample loop from the non-vectorized version.
    scores = X @ W                                          # shape (N, C)
    # subtract the per-row max for numerical stability (log-sum-exp trick).
    scores -= np.max(scores, axis=1, keepdims=True)
    # softmax: exponentiate and normalize per row.
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # shape (N, C)

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Compute the loss.                                                   #
    # 1. Select the probabilities for the correct class for all samples.        #
    #    (Hint: Use advanced integer indexing with np.arange(N) and y)          #
    # 2. Compute the negative log-likelihood for these probabilities.           #
    # 3. Compute the average loss (a scalar) across all samples in the batch.   #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################

    # grab the predicted probability of the correct class for every sample via
    # advanced indexing, then take -log and average over the batch.
    correct_probs = probs[np.arange(N), y]
    loss = float(np.mean(-np.log(correct_probs + 1e-12)))

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################


    #############################################################################
    # TODO: Backward pass: compute gradient dW.                                 #
    # 1. Compute the gradient of the loss with respect to the scores.           #
    # 2. Compute the gradient dW using the chain rule (X.T @ dscores).          #
    # 3. Average the gradient over the batch (divide by N).                     #
    #############################################################################
    # START OF YOUR CODE                                                        #
    #############################################################################

    # dL/dscores for softmax cross-entropy is (probs - one_hot_labels).
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1.0
    # chain rule with a single matmul: dL/dW = X^T @ dL/dscores, then average.
    dW = X.T @ dscores / N

    #############################################################################
    # END OF YOUR CODE                                                          #
    #############################################################################

    return loss, dW


def tune_perceptron(
    ModelClass,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rates: Iterable[float],
    batch_sizes: Iterable[int],
    *,
    num_iters: int = 500,
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[float, int], Tuple[float, float]], Any, float]:
    """
    Hyperparameter sweep for a LinearPerceptron-like model.

    Parameters
    ----------
    ModelClass : class
        Class with API:
          - __init__(X_init, y_init, **kwargs)   # optional; ok if ignores data
          - train(X, y, learning_rate, num_iters, batch_size, verbose=False) -> loss_history
          - calc_accuracy(X, y) -> float
    X_train, y_train, X_val, y_val : arrays
        Training/validation data.
    learning_rates : iterable of float
    batch_sizes : iterable of int
    num_iters : int
        Iterations per configuration.
    model_kwargs : dict or None
        Extra kwargs to pass to ModelClass.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    results : dict
        {(lr, batch): (train_acc, val_acc)} for each tried combo.
    best_model : object
        The fitted model instance with the highest validation accuracy.
    best_val : float
        The best validation accuracy achieved.
    """

    # Initialization
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    results: Dict[Tuple[float, int], Tuple[float, float]] = {}
    best_val = -1.0
    best_model = None

    ############################################################################
    # TODO: Iterate over all combinations of learning_rates and batch_sizes.   #
    #   For each (lr, batch_size):                                             #
    #     1. Create a new model instance using ModelClass(...).                #
    #     2. Train it using the train() method with the given lr and batch.   #
    #     3. Compute training and validation accuracies using calc_accuracy(). #
    #     4. Store results[(lr, batch_size)] = (train_acc, val_acc).           #
    #     5. Track the best model based on validation accuracy.                #
    #                                                                          #
    # Hints:                                                                   #
    # - Use two nested for-loops: outer over learning_rates, inner over        #
    #   batch_sizes.                                                           #
    # - Use 'verbose' to optionally print current hyperparameters.             #
    # - Make sure to create a *new* model for each configuration (do not       #
    #   re-use a trained one).                                                 #
    ############################################################################
    #                          START OF YOUR CODE                              #
    ############################################################################

    # grid search: try every (learning_rate, batch_size) pair.
    for lr in learning_rates:
        for batch_size in batch_sizes:
            if verbose:
                print(f"training with learning_rate={lr}, batch_size={batch_size}")

            # 1) start fresh: a new model for each configuration so results are
            #    comparable and we do not reuse already-trained weights.
            model = ModelClass(X_train, y_train, **model_kwargs)

            # 2) train on the training set with this configuration.
            #    verbose=False here to keep the sweep output compact; the outer
            #    `verbose` flag only controls the short one-line progress print above.
            model.train(
                X_train, y_train,
                learning_rate=lr,
                num_iters=num_iters,
                batch_size=batch_size,
                verbose=False,
            )

            # 3) evaluate on train and validation sets.
            train_acc = model.calc_accuracy(X_train, y_train)
            val_acc = model.calc_accuracy(X_val, y_val)

            # 4) record the scores for this configuration.
            results[(lr, batch_size)] = (train_acc, val_acc)

            if verbose:
                print(f"  -> train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

            # 5) keep the model with the best validation accuracy seen so far.
            if val_acc > best_val:
                best_val = val_acc
                best_model = model

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return results, best_model, best_val
