# Check for missning values nan
df.isnull().values.any()

# Quickly check for nan/null/missing values in all columns - We will deal with each column separately
df.isnull().sum()

df.head(10)
df.info()
df.describe()
df.shape

# Map 
df['mainroad'] = df['mainroad'].map({'yes':1, 'no':0})

# Check the target column distribution
100*(loan_df['loan_default'].astype('object').value_counts()/len(loan_df.index))

# Fill missing or Nan in Categorical with Unknown Category
loan_df['Employment.Type'].fillna('Unknown', inplace=True)

# Convert to datefield
loan_df['DisbursalDate'] = pd.to_datetime(loan_df['DisbursalDate'], format = "%d-%m-%y")

# Extract weeek day, month year etc from checkin 
cm_combined_dataset['day_of_week_checkin_date'] = cm_combined_dataset['checkin_date'].dt.weekday_name
cm_combined_dataset['day_checkin_date'] = cm_combined_dataset['checkin_date'].dt.day
cm_combined_dataset['week_checkin_date'] = cm_combined_dataset['checkin_date'].dt.week

cm_combined_dataset['month_checkin_date'] = cm_combined_dataset['checkin_date'].dt.month
cm_combined_dataset['month_checkin_date'] = cm_combined_dataset['month_checkin_date'].astype('category')

cm_combined_dataset['year_checkin_date'] = cm_combined_dataset['checkin_date'].dt.year
cm_combined_dataset['weekday_checkin_date'] = cm_combined_dataset['checkin_date'].dt.dayofweek
cm_combined_dataset['weekend_checkin_date'] = cm_combined_dataset['day_of_week_checkin_date'].isin(['Saturday','Sunday']).map({True:1, False:0})


# New column stay days from checkin and chekout
cm_combined_dataset['stay_days'] = (cm_combined_dataset['checkout_date'] - cm_combined_dataset['checkin_date']).dt.days

# Comnert multiple column to categorical
column_to_convert_to_categorical = ['target', 'cp', 'fbs', 'exang', 'restecg', 'slope', 'ca', 'thal']
for col in column_to_convert_to_categorical:
    heart_disease_df[col] = heart_disease_df[col].astype('category')