# -*- coding: utf-8 -*-
"""google coolab integration setup

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Zbtcy4YFoD4cNmsfwSEEpbsFk8kfW4Rs
"""

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

!pip install catboost

import re # regex
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.colab import drive
drive.mount('/content/gdrive')

train = pd.read_csv("gdrive/My Drive/Datascience/ClubMahindra/source_training_dataset.csv")

source_training_dataset = train

categorical_column_names = ['booking_type_code', 'cluster_code', 'main_product_code',
       'member_age_buckets', 'memberid', 'persontravellingid',
       'reservationstatusid_code', 'resort_id', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code', 'season_holidayed_code',
       'state_code_residence', 'state_code_resort', 'day_of_week_checkin_date',
       'month_checkin_date', 'day_of_week_checkout_date',
       'month_checkout_date']

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

dependent_column='amount_spent_per_room_night_scaled'

data_for_cb = source_training_dataset.drop('reservation_id', axis=1)

X = data_for_cb.drop(dependent_column,1)
y = data_for_cb[dependent_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=100)

X_train.info()

categorical_features_indices = [X_train.columns.get_loc(col) for col in categorical_column_names]
print(categorical_features_indices)

#### GridSerachCV has some problem - Trying custom grid search
def runCustomGridSearch(depth, learning_rate, iterations):
    ret_df = pd.DataFrame( columns=['depth', 'lr', 'iter', 'r2_test', 'rmse_test'])
    loopCount = 0
    for aDepth in depth:
        for aLR in learning_rate:
            for aIter in iterations:
                regressor = CatBoostRegressor(depth=aDepth, learning_rate=aLR, iterations=aIter, loss_function='RMSE', task_type = "GPU")
                # Fit model
                regressor.fit(X_train, y_train, cat_features = categorical_features_indices)                
                # Get predictions
                y_pred = regressor.predict(X_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
                r2_test = r2_score(y_test, y_pred)                
                ret_df.loc[loopCount]=[aDepth,aLR,aIter,r2_test,rmse_test]
                loopCount = loopCount + 1
                print('loopCount='+str(loopCount))
    return ret_df

result = runCustomGridSearch([10,12,14], [0.1], [1000])
result

!git clone https://github.com/KeshavShetty/ds.git
git pull "https://github.com/KeshavShetty/ds.git"

from KUtils.chartil import chartil

chartil.uni_category_barchart(source_training_dataset, 'cluster_code')

chartil.uni_continuous_boxplot(source_training_dataset, 'amount_spent_per_room_night_scaled')

chartil.multi_continuous_continuous_category_scatterplot(source_training_dataset, 'amount_spent_per_room_night_scaled', 'advanced_booking_days', 'cluster_code')