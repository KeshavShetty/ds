import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

cm_train_df = pd.read_csv("D://projects/PythonWS/ClubMahindra//train.csv")
cm_test_df = pd.read_csv("D://projects/PythonWS/ClubMahindra//test.csv")

cm_train_df.info()

# Quickly check for nan/null/missing values in all columns - We will deal with each column separately
cm_train_df.isnull().sum() # season_holidayed_code 114, state_code_residence 4764 missing
cm_test_df.isnull().sum()  # season_holidayed_code 35, state_code_residence 2260 missing

# Number of unique values in each column (Check in both Train and Test for missing categorial label in any)
cm_train_df_unique_vals = {x: len(cm_train_df[x].unique()) for x in cm_train_df.columns}
cm_train_df_unique_vals

cm_test_df_unique_vals = {x: len(cm_test_df[x].unique()) for x in cm_test_df.columns}
cm_test_df_unique_vals

# Custom function from Kutils -- Take from https://github.com/KeshavShetty/ds
from KUtils.chartil import chartil 


# Data Understanding and Preparation

# Use unified approach for Train and Test dataset while data preparation
cm_test_df['amount_spent_per_room_night_scaled'] = 0 # Any value - Doesn't matter  - Just balancing column count b/w train & test
## Add separator column for train and test data set - To use to filter before modeliing
cm_train_df['dataset_type'] = 'train'
cm_test_df['dataset_type'] = 'test'

cm_combined_dataset = pd.concat([cm_train_df, cm_test_df])
cm_combined_dataset.info()

# Lets go thru each column

### (A) Data understanding, Preparation, addning new features etc 
# Outlier treatment will done at later stage before modeling as few algos accomodates Outliers


## A.1. reservation_id is just a sequnce or unique number for rows - Ignore it for EDA

## A.2 booking_date
cm_combined_dataset['booking_date'].head(10)
# Convert to type datefield
cm_combined_dataset['booking_date'] = pd.to_datetime(cm_combined_dataset['booking_date'], format = "%d/%m/%y")
# Use this feature to calculate advance booking days and drop this column 

## A.3 checkin_date
cm_combined_dataset['checkin_date'].head(10)
# Convert to type datefield
cm_combined_dataset['checkin_date'] = pd.to_datetime(cm_combined_dataset['checkin_date'], format = "%d/%m/%y")
cm_combined_dataset['checkin_date'].describe() # Date between 08/03/2012 01/03/2019
cm_combined_dataset['checkin_date'].head()

# New column days gap between booking and checkin
cm_combined_dataset['advanced_booking_days'] = (cm_combined_dataset['checkin_date'] - cm_combined_dataset['booking_date']).dt.days
# Some negative numbers in new column - How can you checkin before booking?
# Replace negative values with 0
cm_combined_dataset['advanced_booking_days'][cm_combined_dataset['advanced_booking_days'] < 0] = 0
sum(cm_combined_dataset['advanced_booking_days'] < 0)

# Extract weeek day, month year etc from checkin 
cm_combined_dataset['day_of_week_checkin_date'] = cm_combined_dataset['checkin_date'].dt.weekday_name
cm_combined_dataset['day_checkin_date'] = cm_combined_dataset['checkin_date'].dt.day
cm_combined_dataset['week_checkin_date'] = cm_combined_dataset['checkin_date'].dt.week

cm_combined_dataset['month_checkin_date'] = cm_combined_dataset['checkin_date'].dt.month
cm_combined_dataset['month_checkin_date'] = cm_combined_dataset['month_checkin_date'].astype('category')

cm_combined_dataset['year_checkin_date'] = cm_combined_dataset['checkin_date'].dt.year
cm_combined_dataset['weekday_checkin_date'] = cm_combined_dataset['checkin_date'].dt.dayofweek
cm_combined_dataset['weekend_checkin_date'] = cm_combined_dataset['day_of_week_checkin_date'].isin(['Saturday','Sunday']).map({True:1, False:0})

## A.4  checkout_date
cm_combined_dataset['checkout_date'].head(10)
# Convert to type datefield
cm_combined_dataset['checkout_date'] = pd.to_datetime(cm_combined_dataset['checkout_date'], format = "%d/%m/%y")
cm_combined_dataset['checkout_date'].describe() # Date between 08/03/2012 01/03/2019
cm_combined_dataset['checkout_date'].head()

# New column stay days from checkin and chekout
cm_combined_dataset['stay_days'] = (cm_combined_dataset['checkout_date'] - cm_combined_dataset['checkin_date']).dt.days
cm_combined_dataset['stay_days'].describe() # Looks reasonable

# New columns with weekday/end details
# Extract weeek day, month year etc from checkin 
cm_combined_dataset['day_of_week_checkout_date'] = cm_combined_dataset['checkout_date'].dt.weekday_name
cm_combined_dataset['day_checkout_date'] = cm_combined_dataset['checkout_date'].dt.day
cm_combined_dataset['week_checkout_date'] = cm_combined_dataset['checkout_date'].dt.week

cm_combined_dataset['month_checkout_date'] = cm_combined_dataset['checkout_date'].dt.month
cm_combined_dataset['month_checkout_date'] = cm_combined_dataset['month_checkout_date'].astype('category')
cm_combined_dataset['year_checkout_date'] = cm_combined_dataset['checkout_date'].dt.year
cm_combined_dataset['weekday_checkout_date'] = cm_combined_dataset['checkout_date'].dt.dayofweek
cm_combined_dataset['weekend_checkout_date'] = cm_combined_dataset['day_of_week_checkout_date'].isin(['Saturday','Sunday']).map({True:1, False:0})

# Todo: New column Number of bank holidays b/w checkin and checkout
def weekend_count(start, end):
  return np.busday_count(pd.to_datetime(start), pd.to_datetime(end) + pd.Timedelta(days=1), weekmask='Sat Sun')

cm_combined_dataset['weekend_count_during_stay'] = np.vectorize(weekend_count)(cm_combined_dataset['checkin_date'], cm_combined_dataset['checkout_date'])

## A.5 channel_code - Looks like cartegorical data with 3 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'channel_code')
# Convert to categorical type
cm_combined_dataset['channel_code'] = cm_combined_dataset['channel_code'].astype('category')

## A.6 main_product_code - Looks like cartegorical data with 5 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'main_product_code')
# Convert to categorical type
cm_combined_dataset['main_product_code'] = cm_combined_dataset['main_product_code'].astype('category')

## A.7 numberofadults - min:0, max:32!!!, mean: 3.2 
cm_combined_dataset['numberofadults'].describe()
chartil.uni_continuous_boxplot(cm_combined_dataset, 'numberofadults')

## A.8 numberofchildren
cm_combined_dataset['numberofchildren'].describe()
# Todo: There is something odd - Number of adults + Childern doesnt match travelling persons

# New column total guests
cm_combined_dataset['total_numberofguests'] = cm_combined_dataset['numberofadults'] + cm_combined_dataset['numberofchildren']
cm_combined_dataset['total_numberofguests'].describe()

## A.9 persontravellingid - Looks like categorical variable with 6 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'persontravellingid')
# Convert to categorical type
cm_combined_dataset['persontravellingid'] = cm_combined_dataset['persontravellingid'].astype('category')

## A.10 - resort_region_code - Looks like categorical variable with 3 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'resort_region_code')
# Convert to categorical type
cm_combined_dataset['resort_region_code'] = cm_combined_dataset['resort_region_code'].astype('category')

## A.11 resort_type_code - Looks like categorical variable with 7 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'resort_type_code')
# Convert to categorical type
cm_combined_dataset['resort_type_code'] = cm_combined_dataset['resort_type_code'].astype('category')

## A.12 room_type_booked_code - Looks like categorical variable with 6 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'room_type_booked_code')
# Convert to categorical type
cm_combined_dataset['room_type_booked_code'] = cm_combined_dataset['room_type_booked_code'].astype('category')

## A.13 roomnights
cm_combined_dataset['roomnights'].describe()
# Some negative - cap it to 0
cm_combined_dataset['roomnights'][cm_combined_dataset['roomnights'] < 0] = 0

## A.14 season_holidayed_code - Looks like categorical variable with 5 unique values
# Convert to categorical type
cm_combined_dataset['season_holidayed_code'] = cm_combined_dataset['season_holidayed_code'].astype('category')
chartil.uni_category_barchart(cm_combined_dataset, 'season_holidayed_code')

## A.15 state_code_residence - Looks like categorical variable with 38 unique values
# Convert to categorical type
cm_combined_dataset['state_code_residence'] = cm_combined_dataset['state_code_residence'].astype('category')
chartil.uni_category_barchart(cm_combined_dataset, 'state_code_residence')

## A.16 state_code_resort - Looks like categorical variable with 11 unique values
chartil.uni_category_barchart(cm_combined_dataset, 'state_code_resort')
# Convert to categorical type
cm_combined_dataset['state_code_resort'] = cm_combined_dataset['state_code_resort'].astype('category')


## A.17 total_pax - Total persons travelling
cm_combined_dataset['total_pax'].describe()

## A.18 member_age_buckets - categorical variable with 10 unique values 
chartil.uni_category_barchart(cm_combined_dataset, 'member_age_buckets')
# Convert to categorical type
cm_combined_dataset['member_age_buckets'] = cm_combined_dataset['member_age_buckets'].astype('category')


## A.19 booking_type_code - categorical variable with 2 unique values 
chartil.uni_category_barchart(cm_combined_dataset, 'booking_type_code')
# Convert to categorical type
cm_combined_dataset['booking_type_code'] = cm_combined_dataset['booking_type_code'].astype('category')

## A.20 - memberid - 101327 members 
chartil.uni_category_barchart(cm_combined_dataset, 'memberid', limit_bars_count_to=10)
def calculatePreviousTravelPoint(memberId, beforeDate) :
    #print(type(memberId))
    #print(type(pd.to_datetime(beforeDate)))
    return  cm_combined_dataset[
            (cm_combined_dataset['memberid']==memberId) &
            (cm_combined_dataset['checkin_date']<pd.to_datetime(beforeDate) )].count()[0]
    
# past_previlage_points = np.vectorize(calculatePreviousTravelPoint)(cm_combined_dataset['memberid'], cm_combined_dataset['checkin_date'])
# cm_combined_dataset['past_previlage_points'] = past_previlage_points

## A.21 cluster_code - categorical variable with 6 unique values 
chartil.uni_category_barchart(cm_combined_dataset, 'cluster_code')
# Convert to categorical type
cm_combined_dataset['cluster_code'] = cm_combined_dataset['cluster_code'].astype('category')

## A.22 - reservationstatusid_code - categorical variable with 4 unique values 
chartil.uni_category_barchart(cm_combined_dataset, 'reservationstatusid_code')
# Convert to categorical type
cm_combined_dataset['reservationstatusid_code'] = cm_combined_dataset['reservationstatusid_code'].astype('category')

## A.23 - resort_id - categorical variable with 32 unique values 
chartil.uni_category_barchart(cm_combined_dataset, 'resort_id')
# Convert to categorical type
cm_combined_dataset['resort_id'] = cm_combined_dataset['resort_id'].astype('category')


## A.24 - amount_spent_per_room_night_scaled - It is target column - use from train data 
chartil.uni_continuous_boxplot(cm_train_df, 'amount_spent_per_room_night_scaled')

# Actual rooms occuoied
cm_combined_dataset['actual_rooms_occupied'] = cm_combined_dataset['roomnights']/cm_combined_dataset['stay_days']

# Extract resort_booking_share - Indicates demand per resort
groupByResort_id = cm_combined_dataset.groupby('resort_id') 

resortBookingCount_df = groupByResort_id.size().to_frame()
resortBookingCount_df['resort_id'] = resortBookingCount_df.index

resortBookingCount_df.rename(columns={0:'resort_booking_share'}, inplace=True)
resortBookingCount_df['resort_booking_share'] = resortBookingCount_df['resort_booking_share']*100/len(cm_combined_dataset)

cm_combined_dataset = pd.merge(cm_combined_dataset, resortBookingCount_df, on=['resort_id'], how='inner')

# Extract resort_room_demand_share - Indicates demand per room within resort
resort_rooms_occupied_df = groupByResort_id.sum()
resort_rooms_occupied_df['resort_id'] = resort_rooms_occupied_df.index

resort_rooms_occupied_df['avg_actual_rooms_occupied'] = resort_rooms_occupied_df['actual_rooms_occupied']/(7*52) # For 7 years 52 weeks
cm_combined_dataset = pd.merge(cm_combined_dataset, resort_rooms_occupied_df[['resort_id', 'avg_actual_rooms_occupied']], on=['resort_id'], how='inner')

## B. EDA - explore target variable with other features
eda_train = cm_combined_dataset[(cm_combined_dataset['dataset_type']=='train')]
eda_train.info()
{x: len(eda_train[x].unique()) for x in eda_train.columns} # Print unique values


## B.1 amount_spent_per_room_night_scaled
eda_train['numberofadults'].describe()
chartil.uni_continuous_boxplot(eda_train, 'numberofadults')


######################################################################################################################   
######################################## Do some detailed EDA If time permits ######################################## 
######################################################################################################################   

cm_combined_dataset.info()
cm_combined_dataset.isnull().sum() 

# Replace missing values with mode of respective columns (season_holidayed_code and state_code_residence)
# Todo: Use grouping technique for next data preparation iteration
sum(cm_combined_dataset['season_holidayed_code'].isnull()) 
sum(cm_combined_dataset['state_code_residence'].isnull()) 

# Use mode to fill these small missing values
cm_combined_dataset['season_holidayed_code'].fillna(cm_combined_dataset['season_holidayed_code'].mode()[0], inplace=True)
cm_combined_dataset['state_code_residence'].fillna(cm_combined_dataset['state_code_residence'].mode()[0], inplace=True)

cm_combined_dataset.info()

# Drop non required column for modelling
cm_combined_dataset = cm_combined_dataset.drop("booking_date", axis=1)
cm_combined_dataset = cm_combined_dataset.drop("checkin_date", axis=1)
cm_combined_dataset = cm_combined_dataset.drop("checkout_date", axis=1)
#cm_combined_dataset = cm_combined_dataset.drop("memberid", axis=1)

# select all categorical variables to apply lable encoder
df_categorical = cm_combined_dataset.select_dtypes(include=['object', 'category'])

df_categorical = df_categorical.drop('reservation_id', axis=1)
df_categorical = df_categorical.drop('dataset_type', axis=1)
categorical_column_names = df_categorical.columns
df_categorical.head()


# apply Label encoder to df_categorical
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

# concat df_categorical with original df
cm_combined_dataset = cm_combined_dataset.drop(categorical_column_names, axis=1)
cm_combined_dataset = pd.concat([cm_combined_dataset, df_categorical], axis=1)

# Clean memory
del df_categorical
del cm_train_df
del cm_test_df
del cm_train_df_unique_vals
del cm_test_df_unique_vals
del eda_train
del resortBookingCount_df

# Finally data ready - Separate train and test dataset
# Separate train and external test based on dummy column created at the begining
source_training_dataset = cm_combined_dataset[(cm_combined_dataset['dataset_type']=='train')]
source_external_test_dataset = cm_combined_dataset[(cm_combined_dataset['dataset_type']=='test')]

# Drop the column dataset_type which was introduced to separate train and test in combined dataset
source_training_dataset = source_training_dataset.drop("dataset_type", axis=1)
source_external_test_dataset = source_external_test_dataset.drop("dataset_type", axis=1)
source_external_test_dataset = source_external_test_dataset.drop("amount_spent_per_room_night_scaled", axis=1)

source_training_dataset.info()

################################################################################################################
################################                                                ################################
################################                 Model Building                 ################################
################################                                                ################################
################################################################################################################


## After trying various techniques including Linear Regression, Lasso, XGBM only CatBoostRegressor result found acceptable.
## Only that part of code included further.


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
                regressor = CatBoostRegressor(depth=aDepth, learning_rate=aLR, iterations=aIter, loss_function='RMSE')
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
    
result = runCustomGridSearch([10,12,14], [0.05, 0.1], [1000])

res = runCustomGridSearch([10], [0.05], [10000])
res[:1]

# Final best catboost model obtained with 
regressor = CatBoostRegressor(depth=6, learning_rate=0.05, iterations=3000, loss_function='RMSE')
# Fit model
categorical_features_indices_of_external_test = [X_train.columns.get_loc(col) for col in categorical_column_names]

regressor.fit(X_train, y_train, cat_features = categorical_features_indices_of_external_test)

               
# Get predictions
external_test_pred = regressor.predict(source_external_test_dataset.drop(['reservation_id'], axis=1))  

source_external_test_dataset['amount_spent_per_room_night_scaled'] = external_test_pred # amount_spent_per_room_night_scaled

source_external_test_dataset.to_csv("D:\\projects\\PythonWS\\ClubMahindra\\fsubmission.csv", columns=['reservation_id','amount_spent_per_room_night_scaled'], index=False)


source_training_dataset.to_csv("D:\\projects\\PythonWS\\ClubMahindra\\source_training_dataset.csv", index=False)
categorical_column_names



regressor = CatBoostRegressor(learning_rate=0.02, depth=6, iterations=5000, eval_metric="RMSE", verbose=True,bootstrap_type="Bernoulli", task_type = "GPU")
regressor.fit(X_train, y_train, cat_features = categorical_features_indices)

y_pred = regressor.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)