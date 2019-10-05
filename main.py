import google.cloud
from google.cloud import logging as CloudLogging
from google.cloud import storage

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import pickle
import io
import logging
import sys

from models import *

logging_client = CloudLogging.Client()

cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
  'projectId': 'biopred',
})
db = firestore.client()

storage_client = storage.Client()

doc_ref = db.document(sys.argv[1])
doc_dict = doc_ref.get().to_dict()

def fit_model(doc_dict, model):
    return model.predict_input(doc_dict)

if __name__ == "__main__":
    with io.BytesIO() as fp:
        storage_client.get_bucket('biopred-models').blob('model_{}.pkl'.format(sys.argv[1].split('/')[1])).download_to_file(fp)
        fp.seek(0)
        m = pickle.loads(fp.read())

    doc_ref.update({
        "status": "complete",
        "result": fitModel(doc_dict, m)
    })

    sys.exit()
