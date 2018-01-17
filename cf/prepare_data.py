import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dtypes = {
    'id': np.int32,
    'store_nbr': np.int8,
    'item_nbr': np.int32,
    'unit_sales': np.float16,
}

train = pd.read_csv('data/raw/train.csv', dtype=dtypes, parse_dates=[1])
test = pd.read_csv('data/raw/test.csv', dtype=dtypes, parse_dates=[1])

is_test = test.groupby(['store_nbr', 'item_nbr']).apply(lambda x: pd.Series({'is_test': 1})).reset_index()
is_discrete = train.groupby('item_nbr')['unit_sales'].apply(lambda x: np.all((x.values).astype(int) == x.values))
is_discrete = is_discrete.reset_index().rename(columns={'unit_sales': 'is_discrete'})
start_date = train.groupby(['store_nbr', 'item_nbr'])['date'].min()
test_start = test['date'].min()
test['unit_sales'] = -1
print 'loaded data'

df = pd.concat([train, test], axis=0)
del train, test
df['onpromotion'] = df['onpromotion'].map(lambda x: int(x) if not np.isnan(x) else 2).astype(np.int8)
print 'concatenated'

df = df.merge(is_test, how='left', on=['store_nbr', 'item_nbr'])
df = df[df['is_test'] == 1].drop('is_test', axis=1)
print 'filtered test'

# dates
date_range = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
date_idx = range(len(date_range))
dt_to_idx = dict(map(reversed, enumerate(date_range)))
test_start_idx = dt_to_idx[test_start]
df['date'] = df['date'].map(dt_to_idx.get)
missing_dates = list(set(date_idx) - set(df['date']))

# pivot and reindex
df = df.pivot_table(index=['store_nbr', 'item_nbr'], columns='date')
fill = np.zeros([df.shape[0], len(missing_dates)])
fill[:] = np.nan
missing_df = pd.DataFrame(columns=missing_dates, data=fill)

op = pd.concat([df['onpromotion'].reset_index(), missing_df], axis=1).fillna(2)
op = op[['store_nbr', 'item_nbr'] + date_idx]
op = op[date_idx].values.astype(np.int8)

for i in range(op.shape[1]):
    nan_mask = op[:, i] == 2
    p = .2*op[~nan_mask, i].mean()
    if np.isnan(p):
        p = 0
    fill = np.random.binomial(n=1, p=p, size=nan_mask.sum())
    op[nan_mask, i] = fill
print 'nan mean'

uid = pd.concat([df['id'].reset_index(), missing_df], axis=1).fillna(0)
uid = uid[['store_nbr', 'item_nbr'] + date_idx]

df = pd.concat([df['unit_sales'].reset_index(), missing_df], axis=1).fillna(0)
df = df[['store_nbr', 'item_nbr'] + date_idx]

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')

np.save('data/processed/x_raw.npy', df[date_idx].values.astype(np.float16))
np.save('data/processed/onpromotion.npy', op.astype(np.int8))
np.save('data/processed/id.npy', uid[date_idx].values.astype(np.int32))
print 'pivoted'
del op, uid

df[date_idx] = np.log(np.maximum(df[date_idx].values, 0) + 1)
df[date_idx] = df[date_idx].astype(np.float16)
np.save('data/processed/x.npy', df[date_idx].values)

# non-temporal features
start_date = start_date.reset_index().rename(columns={'date': 'start_date'})
start_date['start_date'] = start_date['start_date']
df = df.merge(start_date, how='left', on=['store_nbr', 'item_nbr'])
df['start_date'] = df['start_date'].map(lambda x: dt_to_idx.get(x, test_start_idx))
del start_date

df = df.merge(is_discrete, how='left', on='item_nbr')
df['is_discrete'] = df['is_discrete'].fillna(0).astype(int)
del is_discrete

stores = pd.read_csv('data/raw/stores.csv')
encode_cols = [i for i in stores.columns if i != 'store_nbr']
stores[encode_cols] = stores[encode_cols].apply(lambda x: LabelEncoder().fit_transform(x))
df = df.merge(stores, how='left', on='store_nbr')
del stores

items = pd.read_csv('data/raw/items.csv')
encode_cols = ['family', 'class']
items[encode_cols] = items[encode_cols].apply(lambda x: LabelEncoder().fit_transform(x))
df = df.merge(items, how='left', on='item_nbr')
df['item_nbr'] = LabelEncoder().fit_transform(df['item_nbr'])
del items

features = [
    ('store_nbr', np.int8),
    ('item_nbr', np.int32),
    ('city', np.int8),
    ('state', np.int8),
    ('type', np.int8),
    ('cluster', np.int8),
    ('family', np.int8),
    ('class', np.int16),
    ('perishable', np.int8),
    ('is_discrete', np.int8),
    ('start_date', np.int16)
]

for feature, dtype in features:
    vals = df[feature].values.astype(dtype)
    np.save('data/processed/{}.npy'.format(feature), vals)
print 'finished non-temporal features'


# lags
x = df[date_idx].values

x_lags = [1, 7, 14]
lag_data = np.zeros([x.shape[0], x.shape[1], len(x_lags)], dtype=np.float16)

for i, lag in enumerate(x_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/x_lags.npy', lag_data)
del lag_data

xy_lags = [16, 21, 28, 35, 365/4, 365/2, 365, 365*2, 365*3]
lag_data = np.zeros([x.shape[0], x.shape[1], len(xy_lags)], dtype=np.float16)

for i, lag in enumerate(xy_lags):
    lag_data[:, lag:, i] = x[:, :-lag]

np.save('data/processed/xy_lags.npy', lag_data)
del lag_data

# aggregate time series
groups = [
    ['store_nbr'],
    ['item_nbr'],
    ['family'],
    ['class'],
    ['city'],
    ['state'],
    ['type'],
    ['cluster'],
    ['item_nbr', 'city'],
    ['item_nbr', 'type'],
    ['item_nbr', 'cluster'],
    ['family', 'city'],
    ['family', 'type'],
    ['family', 'cluster'],
    ['store_nbr', 'family'],
    ['store_nbr', 'class']
]

df_idx = df[['store_nbr', 'item_nbr', 'family', 'class', 'city', 'state', 'type', 'cluster']]

aux_ts = np.zeros([df.shape[0], len(date_idx), len(groups)], dtype=np.float16)
for i, group in enumerate(groups):
    print i
    ts = df.groupby(group)[date_idx].mean().reset_index()
    ts = df_idx.merge(ts, how='left', on=group)
    aux_ts[:, :, i] = ts[date_idx].fillna(0).values
np.save('data/processed/ts.npy', aux_ts)
