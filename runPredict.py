import google.cloud
from google.cloud import logging as CloudLogging

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import pickle
import logging
import sys

logging_client = CloudLogging.Client()
logging.warning(sys.argv[1])

cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
  'projectId': 'biopred',
})
db = firestore.client()

doc_ref = db.document(sys.argv[1])
doc_dict = doc_ref.to_dict()

typ1 = None
typ2 = None
typ3 = doc_dict['jobs']

if doct_dict['item1']['itemType'] == 0:
    typ1 = 'PROTEIN'
elif doct_dict['item1']['itemType'] == 1:
    typ1 = 'DNA'
elif doct_dict['item1']['itemType'] == 2:
    typ1 = 'RNA'
elif doct_dict['item1']['itemType'] == 3:
    typ1 = 'NA'
elif doct_dict['item1']['itemType'] == 4:
    typ1 = 'LIGAND'

if 'item2' in doc_dict:
    if doct_dict['item2']['itemType'] == 0:
        typ2 = 'PROTEIN'
    elif doct_dict['item2']['itemType'] == 1:
        typ2 = 'DNA'
    elif doct_dict['item2']['itemType'] == 2:
        typ2 = 'RNA'
    elif doct_dict['item2']['itemType'] == 3:
        typ2 = 'NA'
    elif doct_dict['item2']['itemType'] == 4:
        typ2 = 'LIGAND'

if typ1 == 'LIGAND' and typ3[0] == 'T':
    doc_ref.update({
        'status': 'error'
    })
    sys.exit()

if typ2 == 'LIGAND' and typ3[1] == 'T':
    doc_ref.update({
        'status': 'error'
    })
    sys.exit()

model = pickle.loads(
    db.collection('model').select({
        'typ1': typ1,
        'typ2': typ2,
        'typ3': typ3
    }).stream().to_dict()['model']
)

res = model.predict(doc_dict['item1']['sequence'], doc_dict['item2']['sequence']).tolist()
doc_ref.update({
    'status': 'complete',
    'result': res
})

sys.exit()
