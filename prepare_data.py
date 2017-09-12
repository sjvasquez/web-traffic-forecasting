from __future__ import unicode_literals

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_page(x):
    x = x.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]

df = pd.read_csv('data/raw/train_final.csv', encoding='utf-8')
date_cols = [i for i in df.columns if i != 'Page']

df['name'], df['project'], df['access'], df['agent'] = zip(*df['Page'].apply(parse_page))

le = LabelEncoder()
df['project'] = le.fit_transform(df['project'])
df['access'] = le.fit_transform(df['access'])
df['agent'] = le.fit_transform(df['agent'])
df['page_id'] = le.fit_transform(df['Page'])


if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')

df[['page_id', 'Page']].to_csv('data/processed/page_ids.csv', encoding='utf-8', index=False)

np.save('data/processed/is_nan.npy', df[date_cols].isnull().values.astype(int))
np.save('data/processed/data.npy', df[date_cols].fillna(0).values)
np.save('data/processed/project.npy', df['project'].values)
np.save('data/processed/access.npy', df['access'].values)
np.save('data/processed/agent.npy', df['agent'].values)
np.save('data/processed/page_id.npy', df['page_id'].values)
