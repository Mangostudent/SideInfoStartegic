import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import dataclasses
from typing import Literal

from data import Dataset

# --- 1. Model Architecture ---
class BaseModel:
    def __init__(self):
        self.w_A = None; self.w_B = None
    def _prepare_features(self, X, Z=None):
        X_b = np.c_[np.ones(X.shape[0]), X]
        if Z is not None: XZ_b = np.c_[X_b, Z]; return XZ_b, X_b
        return None, X_b

    def predict_A(self, X, Z):
        XZ_b, _ = self._prepare_features(X, Z); logits = XZ_b @ self.w_A
        prediction = np.sign(logits); prediction[prediction == 0] = 1; return prediction
    def predict_B(self, X):
        _, X_b = self._prepare_features(X); logits = X_b @ self.w_B
        prediction = np.sign(logits); prediction[prediction == 0] = 1; return prediction

    def get_strategic_choice_and_prediction(self, X, Z):
        XZ_b, X_b = self._prepare_features(X, Z)
        logits_A = XZ_b @ self.w_A
        logits_B = X_b @ self.w_B
        pred_A = np.sign(logits_A); pred_A[pred_A == 0] = 1
        pred_B = np.sign(logits_B); pred_B[pred_B == 0] = 1
        final_prediction = np.zeros(X.shape[0], dtype=int)
        choice_is_A = np.zeros(X.shape[0], dtype=bool)
        both_predict_pos = (pred_A == 1) & (pred_B == 1)
        final_prediction[both_predict_pos] = 1
        choice_is_A[both_predict_pos] = True
        only_A_predict_pos = (pred_A == 1) & (pred_B == -1)
        final_prediction[only_A_predict_pos] = 1
        choice_is_A[only_A_predict_pos] = True
        only_B_predict_pos = (pred_A == -1) & (pred_B == 1)
        final_prediction[only_B_predict_pos] = 1
        choice_is_A[only_B_predict_pos] = False
        both_predict_neg = (pred_A == -1) & (pred_B == -1)
        final_prediction[both_predict_neg] = -1
        choice_is_A[both_predict_neg] = True
        return final_prediction, choice_is_A

    def predict_strategic_with_choice(self, X, Z):
        prediction, choice_is_A = self.get_strategic_choice_and_prediction(X, Z)
        color_grid = np.zeros_like(prediction, dtype=int)
        color_grid[(prediction == 1) & choice_is_A] = 2
        color_grid[(prediction == 1) & ~choice_is_A] = 1
        color_grid[(prediction == -1) & ~choice_is_A] = -1
        color_grid[(prediction == -1) & choice_is_A] = -2
        return color_grid

    def evaluate(self, X, Y, Z, strategic=False):
        if not strategic:
            XZ_b, _ = self._prepare_features(X, Z)
            final_logits = XZ_b @ self.w_A
        else:
            final_logits, _ = self.get_strategic_choice_and_prediction(X, Z)
        y_pred = np.sign(final_logits)
        y_pred[y_pred == 0] = 1
        return accuracy_score(Y, y_pred)

    def _logistic_loss(self, y_true, logits): return np.mean(np.log(1 + np.exp(-y_true * logits)))

class StrategicModel(BaseModel):
    def __init__(self, optimizer_method='Nelder-Mead', optimizer_maxiter=500, optimizer_disp=False):
        super().__init__(); self.optimizer_method = optimizer_method; self.optimizer_maxiter = optimizer_maxiter; self.optimizer_disp = optimizer_disp
    def _objective(self, weights, X, Y, Z):
        self.w_A = weights[:4]; self.w_B = weights[4:]; XZ_b, X_b = self._prepare_features(X, Z)
        logits_A = XZ_b @ self.w_A; logits_B = X_b @ self.w_B
        use_A = logits_A >= logits_B
        final_logits = np.where(use_A, logits_A, logits_B); return self._logistic_loss(Y, final_logits)
    def train(self, X, Y, Z):
        initial_weights = np.zeros(7)
        result = minimize(self._objective, initial_weights, args=(X, Y, Z), method=self.optimizer_method, options={'maxiter': self.optimizer_maxiter, 'disp': self.optimizer_disp})
        self.w_A = result.x[:4]; self.w_B = result.x[4:]

class VanillaModel(BaseModel):
    def __init__(self, optimizer_method='Nelder-Mead', optimizer_maxiter=500, optimizer_disp=False):
        super().__init__(); self.optimizer_method = optimizer_method; self.optimizer_maxiter = optimizer_maxiter; self.optimizer_disp = optimizer_disp
    def _objective_A(self, w_A, XZ_b, Y): return self._logistic_loss(Y, XZ_b @ w_A)
    def _objective_B(self, w_B, X_b, Y): return self._logistic_loss(Y, X_b @ w_B)
    def train(self, X, Y, Z):
        XZ_b, X_b = self._prepare_features(X, Z)
        res_A = minimize(self._objective_A, np.zeros(4), args=(XZ_b, Y), method=self.optimizer_method, options={'maxiter': self.optimizer_maxiter, 'disp': self.optimizer_disp})
        self.w_A = res_A.x
        res_B = minimize(self._objective_B, np.zeros(3), args=(X_b, Y), method=self.optimizer_method, options={'maxiter': self.optimizer_maxiter, 'disp': self.optimizer_disp})
        self.w_B = res_B.x

class BayesianModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.yz_pairs = None; self.yz_probabilities = None
        self.centers = None; self.covariances = None

    def learn_parameters_from_dataset(self, dataset: 'Dataset'):
        self.yz_pairs = dataset.yz_pairs
        self.yz_probabilities = dataset.yz_probabilities
        self.centers = dataset.centers
        self.covariances = dataset.covariances
        if hasattr(self, 'verbose') and self.verbose:
            print("\n--- Bayesian Model initialized with true generative parameters (Calculated, not learned) ---")

    def _get_joint_prob(self, y, z):
        for i, pair in enumerate(self.yz_pairs):
            if pair[0] == y and pair[1] == z:
                return self.yz_probabilities[i]
        return 0

    def predict_proba(self, X, Z):
        n_samples = X.shape[0]; prob_Y_is_1 = np.zeros(n_samples)
        for i in range(n_samples):
            x_i, z_i = X[i, :], Z[i]
            mean_y1_z, cov_y1_z = self.centers.get((1, z_i)), self.covariances.get((1, z_i))
            mean_y_neg1_z, cov_y_neg1_z = self.centers.get((-1, z_i)), self.covariances.get((-1, z_i))
            if mean_y1_z is None or mean_y_neg1_z is None:
                prob_Y_is_1[i] = 0.5; continue
            likelihood_y1 = multivariate_normal.pdf(x_i, mean=mean_y1_z, cov=cov_y1_z)
            likelihood_y_neg1 = multivariate_normal.pdf(x_i, mean=mean_y_neg1_z, cov=cov_y_neg1_z)
            joint_prob_y1_z = self._get_joint_prob(1, z_i)
            joint_prob_y_neg1_z = self._get_joint_prob(-1, z_i)
            numerator_y1 = likelihood_y1 * joint_prob_y1_z
            numerator_y_neg1 = likelihood_y_neg1 * joint_prob_y_neg1_z
            denominator = numerator_y1 + numerator_y_neg1
            prob_Y_is_1[i] = 0.5 if denominator < 1e-9 else numerator_y1 / denominator
        return prob_Y_is_1

    def predict(self, X, Z):
        prob_Y_is_1 = self.predict_proba(X, Z)
        return np.where(prob_Y_is_1 >= 0.5, 1, -1)

    def evaluate(self, X, Y, Z):
        y_pred = self.predict(X, Z)
        return accuracy_score(Y, y_pred)
