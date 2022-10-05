from abc import abstractmethod
from pathlib import Path
from typing import Optional

import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tueplots import bundles


class Model:
    def __init__(self):
        ...

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def class_probability(self, x: np.ndarray) -> np.ndarray:
        ...

    def failure_probability(self, x: np.ndarray) -> float:
        return np.mean(self.classify(x)).item()

    def classify(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.class_probability(x) > threshold

    @abstractmethod
    def misclassification_probability(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def plot(self, path: Path):
        ...


def default_regression_kernel(n_inputs: int):
    kernel = GPy.kern.Matern52(n_inputs, ARD=True)
    kernel.lengthscale.constrain_bounded(lower=0, upper=0.2)
    # Upper bound on length scale of Gaussian process. If we know that the rule value will change over a certain input
    # parameter range, we can use this bound to guarantee that the model will predict sufficient uncertainty over this
    # range when we have insufficient training samples.
    kernel.variance.constrain_bounded(0.5, 1)  # Will work as long as the rule values are normalised to [-1, 1]
    # Upper bound on maximum uncertainty in the rule value. Should be set to approximately the range of the rule values.
    # Model will predict this level of uncertainty when training data not present
    return kernel


def default_classification_kernel(n_inputs: int):
    kernel = GPy.kern.Matern52(n_inputs, ARD=True)
    kernel.variance.constrain_fixed(1e5)
    return kernel


class Hierarchical(Model):
    def __init__(self, noise_var: float = 0.005**2):
        super().__init__()
        self.kernel_regression = None
        self.kernel_classification = None
        self.noise_var = noise_var
        self.n_dims: Optional[int] = None
        self.model_classification: Optional[GPy.models.GPClassification] = None
        self.model_regression: Optional[GPy.models.GPRegression] = None

    def train(self, x: np.ndarray, y: np.ndarray):
        self.n_dims = x.shape[-1]
        assert isinstance(self.n_dims, int)
        self.kernel_classification = default_classification_kernel(self.n_dims)
        self.kernel_regression = default_regression_kernel(self.n_dims)
        assert x.shape[0] == x.shape[0]
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        nan_rule_values = ~np.isfinite(y.astype(np.float64))
        self.model_classification = GPy.models.GPClassification(
            x,
            nan_rule_values,
            kernel=self.kernel_classification,
        )

        self.model_regression = GPy.models.GPRegression(
            x[~nan_rule_values[..., 0], ...],
            y[~nan_rule_values[..., 0]],
            kernel=self.kernel_regression,
            noise_var=self.noise_var,
        )
        self.model_regression.likelihood.variance.constrain_bounded(0, self.noise_var)
        self.model_regression.optimize()
        self.model_classification.optimize()

    def class_probability(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(self.model_regression, GPy.models.GPRegression)
        assert isinstance(self.model_classification, GPy.models.GPClassification)
        predict_mean, predict_var = self.model_regression.predict(x, include_likelihood=False)
        p_nan, _ = self.model_classification.predict(x)
        return scipy.stats.norm.cdf(-predict_mean / np.sqrt(predict_var)) * (1 - p_nan)

    def misclassification_probability(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(self.model_regression, GPy.models.GPRegression)
        assert isinstance(self.model_classification, GPy.models.GPClassification)
        probability_of_failure = self.class_probability(x)
        return np.min(
            [probability_of_failure, 1 - probability_of_failure],
            axis=0,
        )

    def plot(self, path: Path):
        assert isinstance(self.model_regression, GPy.models.GPRegression)
        assert isinstance(self.model_classification, GPy.models.GPClassification)
        if self.n_dims == 1:
            _, ax = plt.subplots(dpi=80)
            self.model_regression.plot(ax=ax, lower=15.9, upper=100 - 15.9)
            plt.xlim([0, 1])
            plt.ylim([-1.5, 1.5])
            plt.xlabel("x", fontsize=12)
            plt.ylabel("g(x)", fontsize=12)
            for extension in ["pdf", "png"]:
                plt.savefig(Path(str(path) + f"_reg_gp.{extension}"))
            plt.clf()
            _, ax = plt.subplots(dpi=80)
            self.model_classification.plot(ax=ax, lower=15.9, upper=100 - 15.9)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("NaN probability", fontsize=12)
            for extension in ["pdf", "png"]:
                plt.savefig(Path(str(path) + f"_cls_gp.{extension}"))

        elif self.n_dims == 2:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)

            X, Y = np.meshgrid(x, y)

            Z = np.zeros_like(X)
            Z_var = np.zeros_like(X)
            Z_nan = np.zeros_like(X)
            for i in range(X.shape[0]):
                Z[i, :], Z_var[i, :] = map(
                    lambda out: out.squeeze(), self.model_regression.predict(np.stack([X[i, :], Y[i, :]], axis=-1))
                )
                Z_nan[i, :] = self.model_classification.predict(np.stack([X[i, :], Y[i, :]], axis=-1))[0].squeeze()

            Z[Z_nan > 0.5] = np.nan
            fig = plt.figure()

            ax = fig.add_subplot(111, projection="3d")
            _ = ax.plot_surface(X, Y, Z + np.sqrt(Z_var), label="Real valued")
            _ = ax.plot_surface(X, Y, Z - np.sqrt(Z_var), label="Real valued")
            ax.scatter(self.model_regression.X[:, 0], self.model_regression.X[:, 1], self.model_regression.Y)
            ax.set_zlim(-1.5, 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Adversary start x", fontsize=12)
            ax.set_ylabel("Adversary velocity\n(constant)", fontsize=12)
            ax.set_zlabel("Rule robustness", fontsize=12)

            for extension in ["pdf", "png"]:
                plt.savefig(Path(str(path) + f"_reg_gp.{extension}"), bbox_inches="tight")

            plt.clf()
            fig = plt.figure()

            ax = fig.add_subplot(111, projection="3d")
            _ = ax.plot_surface(X, Y, Z_nan, label="Real valued")
            ax.scatter(
                self.model_classification.X[:, 0],
                self.model_classification.X[:, 1],
                self.model_classification.Y,
            )
            ax.set_zlim(-1.5, 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Adversary start x", fontsize=12)
            ax.set_ylabel("Adversary velocity\n(constant)", fontsize=12)
            ax.set_zlabel("NaN probability", fontsize=12)

            for extension in ["pdf", "png"]:
                plt.savefig(Path(str(path) + f"_cls_gp.{extension}"), bbox_inches="tight")

        else:
            print(f"Can't plot models with {self.n_dims} dims")

        plt.clf()
        plt.close()


class MaskedModel(Model):
    def __init__(self, noise_var: float = 0.005**2, mask_with: float = 1.0):
        self.kernel = None
        self.noise_var = noise_var
        self.model_regression = None
        self.n_dims: Optional[int] = None
        self.mask = mask_with

    def train(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == x.shape[0]
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        self.n_dims = x.shape[-1]
        assert isinstance(self.n_dims, int)
        kernel_regression = default_regression_kernel(self.n_dims)

        nan_rule_values = ~np.isfinite(y.astype(np.float64))

        self.model_regression = GPy.models.GPRegression(
            x,
            np.where(nan_rule_values, self.mask * np.ones_like(y), y),
            kernel=kernel_regression,
            noise_var=self.noise_var,
        )

        assert isinstance(self.model_regression, GPy.models.GPRegression)
        self.model_regression.likelihood.variance.constrain_bounded(0, self.noise_var)
        self.model_regression.optimize()

    def class_probability(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(self.model_regression, GPy.models.GPRegression)
        predict_mean, predict_var = self.model_regression.predict(x, include_likelihood=False)
        return scipy.stats.norm.cdf(-predict_mean / np.sqrt(predict_var))

    def misclassification_probability(self, x: np.ndarray) -> np.ndarray:
        probability_of_failure = self.class_probability(x)
        return np.min(
            [probability_of_failure, 1 - probability_of_failure],
            axis=0,
        )

    def plot(self, path: Path):
        assert isinstance(self.model_regression, GPy.models.GPRegression)
        if self.n_dims == 1:
            _, ax = plt.subplots(dpi=80)
            self.model_regression.plot(ax=ax, lower=15.9, upper=100 - 15.9)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("g(x)", fontsize=12)
            plt.xlim([0, 1])
            plt.ylim([-1.5, 1.5])
        elif self.n_dims == 2:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)

            X, Y = np.meshgrid(x, y)

            Z = np.zeros_like(X)
            Z_var = np.zeros_like(X)
            for i in range(X.shape[0]):
                Z[i, :], Z_var[i, :] = map(
                    lambda out: out.squeeze(), self.model_regression.predict(np.stack([X[i, :], Y[i, :]], axis=-1))
                )
            fig = plt.figure()

            ax = fig.add_subplot(111, projection="3d")
            _ = ax.plot_surface(X, Y, Z + np.sqrt(Z_var), label="Real valued")
            _ = ax.plot_surface(X, Y, Z - np.sqrt(Z_var), label="Real valued")
            ax.scatter(self.model_regression.X[:, 0], self.model_regression.X[:, 1], self.model_regression.Y)
            ax.set_zlim(-1.5, 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Adversary start x", fontsize=12)
            ax.set_ylabel("Adversary velocity\n(constant)", fontsize=12)
            ax.set_zlabel("Rule robustness", fontsize=12)
        else:
            print(f"Can't plot models with {self.n_dims} dims")

        for extension in ["pdf", "png"]:
            plt.savefig(Path(str(path) + f"_gp.{extension}"), bbox_inches="tight")

        plt.clf()
        plt.close()


class ClassificationModel(MaskedModel):
    def __init__(self):
        self.x = None
        self.y = None
        self.model_classification = None

    def plot(self, path: Path):
        assert isinstance(self.model_classification, GPy.models.GPClassification)
        if self.n_dims == 1:
            plt.clf()
            _, ax = plt.subplots(dpi=80)
            N_test = 1000
            X_test = np.linspace(0, 1, N_test).reshape(-1, 1)
            cls_prob = self.class_probability(X_test)
            X_test = X_test.squeeze()
            ax.plot(X_test, cls_prob)
            ax.scatter(
                self.model_classification.X, self.model_classification.Y, color="black", marker="x", linewidth=0.3
            )
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.1, 1.1])
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("Failure Probability", fontsize=12)

        elif self.n_dims == 2:
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)

            X, Y = np.meshgrid(x, y)

            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                Z[i, :] = self.class_probability(np.stack([X[i, :], Y[i, :]], axis=-1))
            fig = plt.figure()

            ax = fig.add_subplot(111, projection="3d")
            _ = ax.plot_surface(X, Y, Z)
            ax.scatter(self.x[:, 0], self.x[:, 1], self.y)
            ax.set_zlim(-1.5, 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Adversary start x", fontsize=12)
            ax.set_ylabel("Adversary velocity\n(constant)", fontsize=12)
            ax.set_zlabel("Failure Probability", fontsize=12)
        else:
            print(f"Can't plot models with {self.n_dims} dims")

        for extension in ["pdf", "png"]:
            plt.savefig(Path(str(path) + f"_gp.{extension}"), bbox_inches="tight")

        plt.clf()
        plt.close()


class MaskedGPClassifier(ClassificationModel):
    def __init__(self, noise_var: float = 0.005**2):
        super().__init__()
        self.kernel = None
        self.noise_var = noise_var
        self.n_dims = None
        self.kernel_classification = None

    def train(self, x: np.ndarray, y: np.ndarray):
        assert x.shape[0] == x.shape[0]
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        self.n_dims = x.shape[-1]

        nan_rule_values = ~np.isfinite(y.astype(np.float64))

        masked_y = np.where(nan_rule_values, np.ones_like(y), y) < 0

        self.x = x
        self.y = masked_y
        self.kernel_classification = GPy.kern.Matern52(self.n_dims, ARD=True)
        assert isinstance(self.kernel_classification, GPy.kern.Kern)
        self.kernel_classification.variance.constrain_fixed(100)
        self.kernel_classification.lengthscale.constrain_bounded(lower=0, upper=0.2)

        self.model_classification = GPy.models.GPClassification(
            x,
            masked_y,
            kernel=self.kernel_classification,
        )
        assert isinstance(self.model_classification, GPy.models.GPClassification)

        self.model_classification.optimize()

    def class_probability(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(self.model_classification, GPy.models.GPClassification)
        p_fail, _ = self.model_classification.predict(x)
        return p_fail[:, 0]
