import pickle
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv('C:/Users/ylou/MLC-Spring 2024/MLC-Week13-16-Group Project-Order Forecast/rohlik-orders-forecasting-challenge/train.csv') ## Load the train dataset

def add_date_features(df): ## Convert date features into numerical features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df.drop('date', axis=1, inplace=True)
    return df

train_df = add_date_features(train_df)

cols_to_drop = ['shutdown', 'mini_shutdown', 'mov_change', 'frankfurt_shutdown', 'precipitation', 'snow', 'user_activity_1', 'user_activity_2'] 
train_df.drop(cols_to_drop, axis=1, inplace=True) ## Drop irrelevant features from train dataset 

label_encoders = {} ## Apply label encoding on the warehouse and holiday_name features
for col in ['warehouse', 'holiday_name']:
    label_encoders[col] = LabelEncoder()
    train_df[col] = label_encoders[col].fit_transform(train_df[col])

def ensure_numeric(df): ## Ensure all features are in numerical format for the model development and prediction
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

train_df = ensure_numeric(train_df)

train_df.drop('id', axis=1, inplace=True) ## Remove "id" feature from train dataset
train_df.drop('blackout', axis=1, inplace=True) ## Remove "blackout" feature from train dataset

X = train_df.drop('orders', axis=1) ## The target variable is "orders" and remove it from the independent variables (X)
y = train_df['orders'] ## Define target variable as "orders" for y

model = XGBRegressor()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
