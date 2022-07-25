import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


class OrdinalClassifier():

    def __init__(self):
        self.clf = RandomForestClassifier()  # this is just a place holder that will be overwritten by inheritance
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, x):
        clfs_predict = {k: self.clfs[k].predict_proba(x) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def set_params(self, **params):
        self.clf.set_params(**params)
        return self

    def get_params(self, deep=True):
        return self.clf.get_params(deep)


class RandomForestOC(OrdinalClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = RandomForestClassifier(**kwargs)


class GradientBoostingOC(OrdinalClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = GradientBoostingClassifier(**kwargs)


class ExtraTreesOC(OrdinalClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = ExtraTreesClassifier(**kwargs)
