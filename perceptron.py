from utils import dot_product


class Perceptron:
    def __init__(self, eta: float = 0.01, n_iter: int = 10) -> None:
        """
        Load the Perceptron
        :param float eta: learning rate
        :param int n_iter: number of iterations
        """
        self.w_ = None
        self.errors_ = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X: list, y: list) -> None:
        """
        Fit the Perceptron
        :param list X:
        :param list y:
        :return: None
        """
        self.w_ = [0] * (len(X[0]) + 1)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def predict(self, X: list) -> int:
        """
        Predict the output of the given input
        :param list X: input
        :return: int - prediction
        """
        if self.net_input(X) >= 0:
            return 1
        else:
            return -1

    def net_input(self, X: list) -> float:
        """
        Calculate the net input
        :param list X: input
        :return: float - net input
        """
        return dot_product(X, self.w_[1:]) + self.w_[0]
