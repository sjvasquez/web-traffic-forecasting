import os

import numpy as np
import pandas as pd


preds_mat = np.load(os.path.join('predictions', 'preds.npy'))
labels_mat = np.load(os.path.join('predictions', 'labels.npy'))
priors_mat = np.load(os.path.join('predictions', 'priors.npy'))
pages_mat = np.load(os.path.join('predictions', 'page_id.npy'))

df = pd.read_csv('data/raw/train_final.csv', encoding='utf-8')
date_cols = [i for i in df.columns if i != 'Page']
datetimes = pd.to_datetime(date_cols, format="%Y/%m/%d")
next_date_cols = pd.date_range(start=datetimes[-1], periods=64, closed='right')
pred_df = pd.DataFrame(preds_mat, columns=next_date_cols)
pred_df['page_id'] = pages_mat

page_id_df = pd.read_csv('data/processed/page_ids.csv', encoding='utf-8')
pred_df = pred_df.merge(page_id_df, how='left', on='page_id')

submit_cols = list(next_date_cols[2:]) + ['Page']
pred_df = pred_df[submit_cols]
pred_df = pd.melt(pred_df, id_vars='Page', var_name='date', value_name='Visits')

keys = pd.read_csv('data/raw/key_2.csv', encoding='utf-8')
keys['date'] = keys.Page.apply(lambda a: a[-10:])
keys['Page'] = keys.Page.apply(lambda a: a[:-11])
keys['date'] = keys['date'].astype('datetime64[ns]')

pred_df = pred_df.merge(keys, how='left', on=['Page', 'date'])
pred_df['Visits'] = pred_df['Visits'].map(np.round).astype(int)
pred_df = pred_df[['Id', 'Visits']].sort_values(by='Id')
pred_df.to_csv('sub.csv', encoding='utf-8', index=False)
