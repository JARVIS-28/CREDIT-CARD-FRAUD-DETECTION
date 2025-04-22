import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def handle_missing_values(df):
    """Fill missing values for numerical and categorical columns."""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def engineer_time_features(df):
    df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_datetime'].dt.hour
    df['trans_day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['trans_month'] = df['trans_datetime'].dt.month
    df['is_weekend'] = df['trans_day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['trans_hour'].isin(list(range(0, 6)) + list(range(22, 24))).astype(int)
    df['dob_dt'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_datetime'] - df['dob_dt']).dt.days // 365
    df.drop(['trans_date_trans_time', 'trans_datetime', 'dob', 'dob_dt'], axis=1, inplace=True)
    return df

def engineer_user_features(df):
    user_col = 'cc_num'
    user_trans_count_map = df[user_col].value_counts().to_dict()
    df['user_trans_count'] = df[user_col].map(user_trans_count_map).fillna(0)
    user_avg_amt_map = df.groupby(user_col)['amt'].mean().to_dict()
    df['user_avg_amt'] = df[user_col].map(user_avg_amt_map).fillna(0)
    df['amt_vs_avg'] = (df['amt'] / df['user_avg_amt']).replace([np.inf, -np.inf], 1).fillna(1)
    df = df.sort_values(by=[user_col, 'unix_time'])
    df['time_since_last_trans'] = df.groupby(user_col)['unix_time'].diff().fillna(0)
    return df.reset_index(drop=True)

def engineer_merchant_features(df):
    merchant_col = 'merchant'
    merchant_fraud_rate = df.groupby(merchant_col)['is_fraud'].mean().to_dict()
    df['merchant_fraud_rate'] = df[merchant_col].map(merchant_fraud_rate).fillna(0)
    merchant_trans_count = df[merchant_col].value_counts().to_dict()
    df['merchant_transaction_count'] = df[merchant_col].map(merchant_trans_count).fillna(0)
    return df

def engineer_geo_features(df):
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    df['distance_from_home'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    return df

def encode_and_scale_features(df, scaler=None, fit_scaler=True):
    # One-hot encode gender and category
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_cols = ['gender', 'category']
    ohe.fit(df[ohe_cols].astype(str))
    ohe_df = pd.DataFrame(ohe.transform(df[ohe_cols].astype(str)), columns=ohe.get_feature_names_out(), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), ohe_df], axis=1)
    # Label encode merchant, city, state, job, zip
    label_cols = ['merchant', 'city', 'state', 'job', 'zip']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    # Drop high-cardinality or identifier columns not needed for modeling
    drop_cols = ['first', 'last', 'street', 'trans_num']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    # Scale numerical features
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in ['is_fraud', 'cc_num']:
        if col in num_cols:
            num_cols.remove(col)
    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df, scaler
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        return df

def preprocess_full(df, scaler=None, fit_scaler=True):
    df = handle_missing_values(df)
    df = engineer_time_features(df)
    df = engineer_user_features(df)
    df = engineer_merchant_features(df)
    df = engineer_geo_features(df)
    if fit_scaler:
        df, scaler = encode_and_scale_features(df, fit_scaler=True)
        return df, scaler
    else:
        df = encode_and_scale_features(df, scaler=scaler, fit_scaler=False)
        return df

# Example usage:


if __name__ == "__main__":
    train_path = "./data/fraudTrain.csv"
    test_path = "./data/fraudTest.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    y_train = df_train['is_fraud']
    y_test = df_test['is_fraud']

    # Do NOT drop 'is_fraud' yet
    df_train_processed, scaler = preprocess_full(df_train, fit_scaler=True)
    df_test_processed = preprocess_full(df_test, scaler=scaler, fit_scaler=False)

    df_train_processed.to_csv("./data/fraudTrain_processed.csv", index=False)
    df_test_processed.to_csv("./data/fraudTest_processed.csv", index=False)
    print("Preprocessing complete. Processed files saved.")