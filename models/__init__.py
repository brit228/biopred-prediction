import pandas as pd
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.models import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, accuracy_score


class AutoEncodeRandomForest:
    def __init__(self, maxSeqLength):
        self._encoder = None
        self._randomForest = None
        self.maxSeqLength = maxSeqLength
        self.ngrams = 0
        self._encode_weights = None
        
    def _genEncodeData(self, x):
        temp = np.zeros((1,self.maxSeqLength+2,28))
        for i,v in enumerate('_'+x+'_'):
            temp[0,i,ord(v)-65 if v!= '_' else 26] = 1.0
        temp[0,i+1:,27] = 1.0
        return temp

    def _genData(self, df, M, encoder):
        out = []
        yout = []
        for c in df.index:
            seq = df.loc[c, 'sequence']
            inter = df.loc[c, 'interaction']
            seqEnc = encoder.predict(self._genEncodeData(seq))[0]
            for d in range(len(seq)):
                ind1 = d - M if d > M else 0
                ind2 = d + M + 1 if d + M < len(seq) else len(seq)
                temp = np.zeros((2*M+1,27))
                for i,v in enumerate((ind1 - d + M) * '_' + seq[ind1:ind2] + (d + M + 1 - ind2) * '_'):
                    temp[i,ord(v)-65 if v != '_' else 26] = 1.0
                out.append(np.concatenate([temp.flatten(), seqEnc], axis=0))
            yout += [float(i) for i in inter]
        return np.array(out), np.array(yout)
    
    def _genDataPredict(self, df, M, encoder):
        out = []
        for c in df.index:
            seq = df.loc[c, 'sequence']
            seqEnc = encoder.predict(self._genEncodeData(seq))[0]
            for d in range(len(seq)):
                ind1 = d - M if d > M else 0
                ind2 = d + M + 1 if d + M < len(seq) else len(seq)
                temp = np.zeros((2*M+1,27))
                for i,v in enumerate((ind1 - d + M) * '_' + seq[ind1:ind2] + (d + M + 1 - ind2) * '_'):
                    temp[i,ord(v)-65 if v != '_' else 26] = 1.0
                out.append(np.concatenate([temp.flatten(), seqEnc], axis=0))
        return np.array(out)
    
    def _genDataPredictSeq(self, seq, M, encoder):
        out = []
        seqEnc = encoder.predict(self._genEncodeData(seq))[0]
        for d in range(len(seq)):
            ind1 = d - M if d > M else 0
            ind2 = d + M + 1 if d + M < len(seq) else len(seq)
            temp = np.zeros((2*M+1,27))
            for i,v in enumerate((ind1 - d + M) * '_' + seq[ind1:ind2] + (d + M + 1 - ind2) * '_'):
                temp[i,ord(v)-65 if v != '_' else 26] = 1.0
            out.append(np.concatenate([temp.flatten(), seqEnc], axis=0))
        return np.array(out)

    def fit(self, samp, layer_size):
        X = np.concatenate(samp['sequence'].apply(self._genEncodeData).values, axis=0)
        input_layer = Input(shape=(X.shape[1],X.shape[2]))
        flatten_layer = Flatten()(input_layer)
        encoding_layer = Dense(layer_size, activation='linear')(flatten_layer)
        out_layer = Dense(X.shape[1]*X.shape[2], activation='softmax')(encoding_layer)
        reshape_layer = Reshape((X.shape[1],X.shape[2]))(out_layer)

        model = Model(inputs=[input_layer], outputs=[reshape_layer])
        self._encoder = Model(inputs=[input_layer], outputs=[encoding_layer])

        model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, X, epochs=25, batch_size=128, shuffle=True, verbose=True)
        self._encode_weights = self._encoder.get_weights()
        
    def fitRandomForest(self, samp, ngrams):
        self.ngrams = ngrams
        x, y = self._genData(samp, ngrams, self._encoder)
        self._randomForest = RandomForestClassifier(
            n_estimators=64,
            n_jobs=-1
        )
        self._randomForest.fit(x, y)
        
    def score(self, samp):        
        x, y = self._genData(samp, self.ngrams, self._encoder)
        return (
            self._randomForest.score(x, y),
            roc_auc_score(y, self._randomForest.predict_proba(x)[:,1]),
            average_precision_score(y, self._randomForest.predict_proba(x)[:,1]),
            x, y, self._randomForest.predict_proba(x)[:,1]
        )
    
    def predict(self, samp):
        x = self._genDataPredict(samp, self.ngrams, self._encoder)
        return self._randomForest.predict(x)
    
    def predict_proba(self, samp):
        x = self._genDataPredict(samp, self.ngrams, self._encoder)
        return self._randomForest.predict_proba(x)
    
    def predict_input(self, doc_dict):
        self._encoder.set_weights(self._encode_weights)
        x = self._genDataPredictSeq(doc_dict['sequence'], self.ngrams, self._encoder)
        res = self._randomForest.predict_proba(x)[:,1]
        out = []
        for s in range(len(doc_dict['sequence'])):
            out.append({
                'resabbrev': doc_dict['sequence'][s],
                'interaction': res[s]
            })
        return out
