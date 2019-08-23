import scipy

class PredictModel:
    def __init__(self, tf1, tf2, model, sub_tf1=None, sub_tf2=None):
        self.tf1 = tf1
        self.tf2 = tf2
        self.sub_tf1 = sub_tf1
        self.sub_tf2 = sub_tf2
        self.model = model

    def fit_tf(self, X1, X2, x1=None, x2=None):
        self.tf1.fit(X1)
        self.tf1.fit(X2)
        if self.sub_tf1:
            self.sub_tf1.fit(x1)
        if self.sub_tf2:
            self.sub_tf2.fit(x2)

    def fit_model(self, X1, X2, y, x1=None, x2=None):
        X1 = self.tf1.transform(X1)
        X2 = self.tf2.transform(X2)
        X = scipy.sparse.hstack([X1, X2])
        if self.sub_tf1:
            x1 = self.sub_tf1.transform(x1)
            X = scipy.sparse.hstack([X, x1])
        if self.sub_tf2:
            x2 = self.sub_tf2.transform(x2)
            X = scipy.sparse.hstack([X, x2])
        self.model.fit(X, y)


class PredictModelTT(PredictModel):
    def __init__(self, tf1, tf2, model, sub_tf1, sub_tf2):
        super().__init__(tf1, tf2, model, sub_tf1, sub_tf2)

    def predict(self, X1, X2):
        def neighbors(seq, index):
            M = 4
            ind1 = 0
            ind2 = len(seq)
            if index - M > ind1:
                ind1 = index - M
            if index + M < ind2:
                ind2 = index + M
            return seq[ind1:ind2]
        
        x1 = [neighbors(X1, i) for i in range(len(X1)) for j in range(len(X2))]
        x2 = [neighbors(X2, j) for i in range(len(X1)) for j in range(len(X2))]
        X1 = [X1] * len(X1) * len(X2)
        X2 = [X2] * len(X1) * len(X2)
        X1 = self.tf1.transform(X1)
        X2 = self.tf2.transform(X2)
        x1 = self.sub_tf1.transform(x1)
        x2 = self.sub_tf2.transform(x2)
        X = scipy.sparse.hstack([X1, X2, x1, x2])
        return self.model.predict_proba(X)[:,1]

class PredictModelTF(PredictModel):
    def __init__(self, tf1, tf2, model, sub_tf1):
        super().__init__(tf1, tf2, model, sub_tf1)

    def predict(self, X1, X2):
        def neighbors(seq, index):
            M = 4
            ind1 = 0
            ind2 = len(seq)
            if index - M > ind1:
                ind1 = index - M
            if index + M < ind2:
                ind2 = index + M
            return seq[ind1:ind2]
        
        x1 = [neighbors(X1, i) for i in range(len(X1))]
        X1 = [X1] * len(X1)
        X2 = [X2] * len(X1)
        X1 = self.tf1.transform(X1)
        X2 = self.tf2.transform(X2)
        x1 = self.sub_tf1.transform(x1)
        X = scipy.sparse.hstack([X1, X2, x1])
        return self.model.predict_proba(X)[:,1]

class PredictModelFF(PredictModel):
    def __init__(self, tf1, tf2, model):
        super().__init__(tf1, tf2, model)

    def predict(self, X1, X2):
        X1 = self.tf1.transform([X1])
        X2 = self.tf2.transform([X2])
        X = scipy.sparse.hstack([X1, X2])
        return self.model.predict_proba(X)[:,1]
