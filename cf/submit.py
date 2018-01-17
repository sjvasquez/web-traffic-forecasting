import pandas as pd
import numpy as np
import os

lengths_mat = np.load(os.path.join('predictions', 'lengths.npy'))
preds_mat = np.load(os.path.join('predictions', 'preds.npy'))
ids_mat = np.load(os.path.join('predictions', 'ids.npy'))
preds_mat[lengths_mat == 0] = 0

df = pd.DataFrame({'id': ids_mat.flatten(), 'unit_sales': preds_mat.flatten()})
df['unit_sales'] = df['unit_sales'].map(np.expm1)

df[['id', 'unit_sales']].to_csv('sub.csv', index=False)
