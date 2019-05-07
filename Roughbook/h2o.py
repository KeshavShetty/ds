
###### Refer https://dzone.com/articles/machine-learning-with-h2o-hands-on-guide-for-data ######

# Install h2o if not already installed
!pip install h2o

import h2o
from h2o.frame import H2OFrame
from h2o.automl import H2OAutoML

# Initialize the server, Will start H2o on localhost at port 54321 i.e: accessible at http://localhost:54321
h2o.init()

# Load the data
train=h2o.import_file("D:\\projects\\PythonWS\\ClubMahindra\\source_training_dataset.csv")

x = train.columns
y = "amount_spent_per_room_night_scaled"
x.remove(y)
x.remove('dataset')
x.remove('reservation_id')
x.remove('memberid')

# Categorical columns
code_cols = ['booking_type_code', 'cluster_code', 'main_product_code',
       'member_age_buckets', 'memberid', 'persontravellingid',
       'reservationstatusid_code', 'resort_id', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code', 'season_holidayed_code',
       'state_code_residence', 'state_code_resort', 'day_of_week_checkin_date',
       'month_checkin_date', 'day_of_week_checkout_date',
       'month_checkout_date'
            ]

# Convert all categorical variavle as factros. 
for col in code_cols:
    train[col] = train[col].asfactor()

# Run H2OAutoML with 30 models
aml = H2OAutoML(max_models=30, seed=42,max_runtime_secs=7200,project_name="ClubMahindra_Version2",max_runtime_secs_per_model=400)
# Train the models
aml.train(x=x, y=y, training_frame=train)

# Convert H2o frame to Pandas dataframe (In Spyder it fails to load or display H2o frames)
pandas_df =  aml.leaderboard.as_data_frame()
pandas_df 
aml.leader

# Todo predict (Us ethe best model to predict on test server)

# Shutdown
h2o.cluster().shutdown()
