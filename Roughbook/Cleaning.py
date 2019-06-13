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

# Comvert string to Numeric and make nan when applicable
telecom['TotalCharges'] = pd.to_numeric(telecom['TotalCharges'], errors='coerce')

	
	
# Reload the library package function
import importlib
importlib.reload(module)

# Set decimal places precision to 3 digit (Gloabl effect)
pd.options.display.float_format = '{:,.3f}'.format

# Merge two dataframe with a primary key
df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID' )

# Remove nan values from TotalCharges
telecom = telecom[~np.isnan(telecom['TotalCharges'])]

# Describe by percentile
num_telecom.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99])

# Select one or few column from dataframe
df.loc[:, ~df.columns.isin(['col1', 'col2'])]

# Sort dataframe by columns
newd_sorted_df = df.sort_values(by=['col1','col2'], ascending=[False,True]) # Sort col1 Descending and col2 by ascending

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# Format decimal numbers
print("Accuracy {0:.3f}, Precision {1:.3f}, Recall {2:.3f}, f1_score {3:.3f}, roc_auc {4:.3f}, Sensitivity {5:.3f}, Specificity {6:.3f}".format(accuracy, precision,recall,f1_score,roc_auc,sensitivity,specificity))
