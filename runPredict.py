import google.cloud
from google.cloud import logging as CloudLogging
from google.cloud import storage

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import numpy as np
import scipy as sp
import scipy.sparse

import pickle
import io
import logging
import sys


def fitModel(x1, x2, m, tf1, tf2, res1, res2, res2All, M=3):
    if res1 and res2 and res2All:
        X = np.array([[x1,x2,x1[i-M if i-M>0 else 0:i+M+1 if i+M+1>len(x1) else len(x1)],x2[j-M if j-M>0 else 0:j+M+1 if j+M+1>len(x2) else len(x2)]] for i in range(len(x1)) for j in range(len(x2))])
        X = sp.sparse.hstack([
            tf1.transform(X[:,0]),
            tf2.transform(X[:,1]),
            tf1.transform(X[:,2]),
            tf2.transform(X[:,3])
        ])
        out = m.predict_proba(X)[:,1].reshape((len(x1),len(x2)))
    elif res1 and not res2 and res2All:
        X = np.array([[x1,x2,x1[i-M if i-M>0 else 0:i+M+1 if i+M+1>len(x1) else len(x1)]] for i in range(len(x1))])
        X = sp.sparse.hstack([
            tf1.transform(X[:,0]),
            tf2.transform(X[:,1]),
            tf1.transform(X[:,2])
        ])
        out = m.predict_proba(X)[:,1].reshape((len(x1),1))
    elif not res1 and not res2 and res2All:
        X = np.array([[x1,x2]])
        X = sp.sparse.hstack([
            tf1.transform(X[:,0]),
            tf2.transform(X[:,1])
        ])
        out = m.predict_proba(X)[:,1].reshape((1,1))
    elif res1 and not res2 and not res2All:
        X = np.array([[x1,x1[i-M if i-M>0 else 0:i+M+1 if i+M+1>len(x1) else len(x1)]] for i in range(len(x1))])
        X = sp.sparse.hstack([
            tf1.transform(X[:,0]),
            tf1.transform(X[:,1])
        ])
        out = m.predict_proba(X)[:,1].reshape((len(x1),1))
    elif not res1 and not res2 and not res2All:
        X = np.array([[x1]])
        X = sp.sparse.hstack([
            tf1.transform(X[:,0])
        ])
        out = m.predict_proba(X)[:,1].reshape((1,1))
    return out.tolist()


logging_client = CloudLogging.Client()
logging.warning(sys.argv[1])

cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
  'projectId': 'biopred',
})
db = firestore.client()

storage_client = storage.Client()

doc_ref = db.document(sys.argv[1])
doc_dict = doc_ref.get().to_dict()

typ1 = doc_dict['item1']['itemType'].lower()
typ2 = doc_dict['item2']['itemType'].lower()
typ3 = doc_dict['predSubSeqItem1']
typ4 = doc_dict['predSubSeqItem2']
typ5 = doc_dict['item2']['searchType'] == 'ALL'

str_suffix = '{}_{}_{}'.format(
    doc_dict['predSubSeqItem1']*1,
    doc_dict['predSubSeqItem2']*1,
    (doc_dict['item2']['searchType'] == 'ALL')*1
)


with io.BytesIO() as fp:
    storage_client.get_bucket('biopred-models').blob('model_{}.pkl'.format(typ1)).download_to_file(fp)
    tf1 = pickle.dumps(fp.read1())
if typ1 != typ2:
    with io.BytesIO() as fp:
        storage_client.get_bucket('biopred-models').blob('model_{}.pkl'.format(typ2)).download_to_file(fp)
        tf2 = pickle.dumps(fp.read1())
else:
    tf2 = tf1

with io.BytesIO() as fp:
    storage_client.get_bucket('biopred-models').blob('model_{}_{}_{}.pkl'.format(typ1, typ2, str_suffix)).download_to_file(fp)
    m = pickle.dumps(fp.read1())

doc_ref.update({
    "status": "complete",
    "result": fitModel(doc_dict['item1']['sequence'], doc_dict['item2']['sequence'], m, tf1, tf2, typ3, typ4, typ5)
})

sys.exit()
