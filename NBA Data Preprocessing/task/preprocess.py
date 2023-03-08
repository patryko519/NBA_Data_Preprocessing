import numpy as np
import pandas as pd
import os
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(path):
    df = pd.read_csv(path)
    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'].fillna('No Team', inplace=True)
    df['height'] = df['height'].apply(lambda x: x[-4::].strip()).astype('float')
    df['weight'] = df['weight'].apply(lambda x: x[x.index('/')+1:-3].strip()).astype('float')
    df['salary'] = df['salary'].apply(lambda x: x[1::]).astype('float')
    df['country'] = df['country'].apply(lambda x: 'USA' if x == 'USA' else 'Not-USA')
    df['draft_round'] = df['draft_round'].apply(lambda x: '0' if x == 'Undrafted' else x)

    return df


def feature_data(data_frame):
    data_frame['version'] = data_frame['version'].apply(lambda x: (2020 if x[-2:] == 20 else 2021))
    data_frame['version'] = pd.to_datetime(data_frame['version'], format='%Y')
    data_frame['age'] = (data_frame['version'] - data_frame['b_day']).astype('timedelta64[Y]')
    data_frame['experience'] = (data_frame['version'] - data_frame['draft_year']).astype('timedelta64[Y]') - 1
    data_frame['bmi'] = data_frame['weight'] / data_frame['height'] ** 2
    data_frame = data_frame.drop(['version', 'b_day', 'draft_year', 'weight', 'height'], axis=1)

    cols = []
    for col in data_frame.columns:
        if data_frame[col].dtype == "object" or data_frame[col].dtype == 'datetime64[ns]':
            cols.append(col)
    data_frame.drop(data_frame.loc[:, cols].loc[:, data_frame.nunique() > 50], axis=1, inplace=True)
    return data_frame


def multicol_data(data_frame):
    return data_frame.drop(['age'], axis=1)


def transform_data(data_frame):
    y = data_frame['salary']
    num_feat_df = data_frame.select_dtypes('number')
    num_feat_df = num_feat_df.drop(['salary'], axis=1)
    scaler = StandardScaler()
    scaled_num_feat = scaler.fit_transform(num_feat_df)
    num_feat_df = pd.DataFrame(scaled_num_feat, index=num_feat_df.index, columns=num_feat_df.columns)
    cat_feat_df = data_frame.select_dtypes('object')
    hot_encoder = OneHotEncoder()
    scaled_cat_feat = hot_encoder.fit_transform(cat_feat_df)
    col_name = np.concatenate(hot_encoder.categories_).ravel().tolist()
    cat_feat_df = pd.DataFrame.sparse.from_spmatrix(scaled_cat_feat, columns=col_name)
    final_df = pd.concat([num_feat_df, cat_feat_df], axis=1, join='inner')
    return final_df, y


transform_data(multicol_data(feature_data(clean_data(data_path))))