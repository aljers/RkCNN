import numpy as np
import pandas as pd
import matplotlib as plt
import random
from scipy.stats import mode
from math import dist
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.validation import _num_samples
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.base import ClassifierMixin
from sklearn.utils.extmath import weighted_mode

class kConditionNN(KNeighborsMixin, ClassifierMixin, NeighborsBase):
    def __init__(
                self,
                n_neighbors=5,
                *,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=None,
                smoothing=1):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights
        self.model_list = []
        self.smoothing = smoothing
    def seperation_score(self,X , y):
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        centers = [np.mean(X[y==i],axis=0) for i in set(y)]
        bv = sum([dist(i, np.mean(X, axis=0)) for i in centers])
        wv = 0
        for i in range(len(centers)):
            Nc = X[y==i].shape[0]
            for _, j in X[y == list(set(y))[i]].iterrows():
                wv += dist(centers[i], j)/Nc
        return bv/wv

    def fit(self, X, y):

        return self._fit(X, y)

    def predict(self, X):
        classes_ = self.classes_
        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        temp_X = self._fit_X
        temp_y = self._y
        temp = np.empty((n_queries,n_outputs))
        for index, classes_k in enumerate(classes_):
            temp_model = KNeighborsClassifier(n_neighbors=self.n_neighbors
                                              ).fit(temp_X[temp_y==classes_k],
                                                    temp_y[temp_y==classes_k])
            temp_dist, temp_ind = temp_model.kneighbors(X)
            temp[:,index] = temp_dist[:, self.n_neighbors-1]
        y_pred = np.argmin(temp, axis=1)
        return y_pred

    def predict_proba(self, X):

        classes_ = self.classes_
        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        smooth = -self.smoothing/n_outputs
        temp_X = self._fit_X
        temp_y = self._y
        temp = np.empty((n_queries,n_outputs)
                        )
        for index, classes_k in enumerate(classes_):
            #self._fit_X = temp_X[temp_y==classes_k]
            #self._y = temp_y[temp_y==classes_k]
            temp_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            temp_model.fit(temp_X[temp_y==classes_k],temp_y[temp_y==classes_k])
            temp_dist, temp_ind = temp_model.kneighbors(X)
            temp[:,index] = temp_dist[:, self.n_neighbors-1]
        y_pred_proba = pow(temp, smooth)/pow(temp, smooth).sum(axis=1, keepdims=True)
        if np.isnan(y_pred_proba).any():
            y_pred_proba[np.isnan(y_pred_proba)] = 1
            y_pred_proba = y_pred_proba/y_pred_proba.sum(axis=1, keepdims=True)
        #self._fit_X = temp_X
        #self._y = temp_y
        return y_pred_proba

class RandomkCNN(kConditionNN):
    def __init__(
                self,
                n_neighbors,
                m,
                r,
                h,
                *,
                random_state = None,
                use_score=True,
                smoothing = 1):
        super().__init__(
            n_neighbors=n_neighbors
        )
        self.m = m
        self.r = r
        self.h = h
        self.n_neighbors = n_neighbors
        self.use_score = use_score
        self.model_list = []
        self.score = []
        self.random_state = random_state
        self.smoothing = smoothing

    def feature_sampling(self, data, seed = None):
        return data.sample(n=self.m, replace=False, axis=1, random_state = seed)

    def fit(self, X, y):
        self.model_list = []
        self.score = []
        if not self.random_state:
            self.random_state = np.random.randint(100000)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        centers = [np.mean(X[y==i],axis=0) for i in set(y)]
        A = []
        for i in centers:
            A.append([(px - qx) ** 2.0 for px, qx in zip(i, np.mean(X,axis=0))])
        B = []
        Nc = np.array([])
        for i in range(len(centers)):
            Nc = np.append(Nc,[1/X[y==i].shape[0]]*X[y==i].shape[0])
            for _, j in X[y == i].iterrows():
                B.append([(px - qx) ** 2.0  for px, qx in zip(centers[i],j)])
        for i in range(self.r):
            X_sub = self.feature_sampling(X, seed = self.random_state + i)
            model = kConditionNN(n_neighbors=self.n_neighbors, smoothing=self.smoothing)
            model.fit(X_sub, y)
            self.model_list.append(model)
            a = np.array([1 if i in X_sub.columns else 0 for i in list(X.columns)])
            aA = np.sqrt(np.array(A).dot(a)).sum()
            aB = np.sqrt(np.array(B).dot(a)*Nc).sum()
            self.score.append(aA/aB)
        self.score = np.array(self.score)
        return self._fit(X,y)

    def predict_prob(self, X):
        # prediction using estimated probabilities from kCNN,
        # and conduct the estimated class of RkCNN by the averaging component probabilities.
        temp_model = self.model_list
        temp_score = self.score
        model_index = np.where(self.score > np.sort(self.score)[-self.h])
        self.model_list = list(np.array(self.model_list)[model_index])
        self.score = self.score[model_index]
        res = []
        for model in self.model_list:
            X_sub = X.loc[:, model.feature_names_in_]
            res.append(model.predict_proba(X_sub))

        if len(self.model_list) > 0:
            if self.use_score == True:
                weight = self.score/self.score.sum(axis=0, keepdims=True)
                y_pred = np.average(res, weights=weight, axis=0)
            else:
                y_pred = np.array(res).mean(axis=0)
            self.model_list = temp_model
            self.score = temp_score
            return y_pred
        else:
            self.model_list = temp_model
            self.score = temp_score
            return 'No model is fitted'

    def predict(self, X):
        # prediction using estimated probabilities from kCNN,
        # and conduct the estimated class of RkCNN by the maximum probability.

        temp_model = self.model_list
        temp_score = self.score
        model_index = np.where(self.score > np.sort(self.score)[-self.h])
        self.model_list = list(np.array(self.model_list)[model_index])
        self.score = self.score[model_index]
        res = []
        for model in self.model_list:
            X_sub = X.loc[:, model.feature_names_in_]
            res.append(model.predict(X_sub))

        if len(self.model_list) > 0:
            if self.use_score == True:
                weight = self.score / self.score.sum(axis=0, keepdims=True)
                y_pred = np.argmax(np.average(res, weights=weight, axis=0), axis=1)
            else:
                y_pred = np.argmax(np.array(res).mean(axis=0), axis=1)
            self.model_list = temp_model
            self.score = temp_score
            return y_pred
        else:
            self.model_list = temp_model
            self.score = temp_score
            return 'No model is fitted'
