import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Custom function from Kutils
from KUtils import chartil # chartil.py include in packages

loan_df = pd.read_csv("D:\\projects\\PythonWS\\LTFS\\train.csv")
loan_df.shape # (233154, 42)
loan_df_test = pd.read_csv("D:\\projects\\PythonWS\\LTFS\\test_bqCt9Pv.csv")
loan_df_test.shape # (112392, 42)

loan_df.info()

# Check the target column distribution
100*(loan_df['loan_default'].astype('object').value_counts()/len(loan_df.index))
# Found moderately balanced
# 0    78.292888
# 1    21.707112

# Quickly check for nan/null/missing values in all columns - We will deal with each column separately
loan_df.isnull().sum() # Only Employment.Type has some missing values, Check it in test data as well
loan_df_test.isnull().sum() # Employment.Type - Around 7661 in train and 3443 missing in test 



# Number of unique values in each column (Check in both Train and Test for missing categorial label in any)
loan_df_unique_vals = {x: len(loan_df[x].unique()) for x in loan_df.columns}
loan_df_unique_vals
loan_df_test_unique_vals = {x: len(loan_df_test[x].unique()) for x in loan_df_test.columns}
loan_df_test_unique_vals 
# There are some differences in category labels in different column b/w train and test - Better approach is combine train and test dataset durtng data preparation and separate before the modelling
# Else Trained model will fail to map all posisble values because of missing unique values in test dataset categorical columns (Since we have unique columsn we should be fine to separate )
# Add dummy target column for test data set
loan_df_test['loan_default'] = 1 # Any value - Doesn't matter  - Just balancing column count b/w train & test
## Add separator column for train and test data set - To use to filter before modeliing
loan_df['dataset_type'] = 'train'
loan_df_test['dataset_type'] = 'test'

train_test_combined_dataset = pd.concat([loan_df, loan_df_test])
# Continue the analysis on train dataset, any changes, feature extraction, scaling etc to be applied parallely to combined dataset

# Cleam memory from non required dataset
del loan_df_test
del loan_df_unique_vals
del loan_df_test_unique_vals


# Some useful functions
def cap_outliers(df, column_to_treat):
    q1 = loan_df[column_to_treat].quantile(0.25)
    q3 = loan_df[column_to_treat].quantile(0.75)
    iqr = q3 - q1
    lower_value = q1 - 1.5*iqr
    upper_value = q3 + 1.5*iqr
    df[column_to_treat][df[column_to_treat]<lower_value] = lower_value
    df[column_to_treat][df[column_to_treat]>upper_value] = upper_value

# For AVG.ACCOUNT.AGE and CREDIT.HISTORY
word_separator = ' '
first_prefix = 'yrs'
second_prefix = 'mon'
first_multuplier = 12
second_multiplier = 1

def parseYrsMonStr(input_str):
    all_words = input_str.split(word_separator)
    if len(all_words)>1:
        return (int(all_words[0].split(first_prefix)[0])*first_multuplier + int(all_words[1].split(second_prefix)[0])*second_multiplier)/12
    else:
        return np.nan

# Lets go thru each column
        
### (A) Data understanding, Preparation, addning new features etc 
# Outlier treatment will done at later stage before modeling as few algos accomodates Outliers

## A.0 First check the target column loan_default distribution
chartil.uni_category_barchart(loan_df, 'loan_default')
100*(loan_df['loan_default'].astype('object').value_counts()/len(loan_df.index)) # 0->78.29% and 1->21.70% approximately
# Convert to a categorical and check the bar chart
loan_df['loan_default'] = loan_df['loan_default'].astype('category')
train_test_combined_dataset['loan_default'] = train_test_combined_dataset['loan_default'].astype('category') # Apply changes on loan_df to combined dataset

loan_df['loan_default'].describe()
loan_df['loan_default'].dtype

## A.1. Unique Id is just a sequnce of numbers - Ignore it for EDA

## A.2. disbursed_amount is a numeric # Ranging from 13320 to 990572 looks like has some outlier 
loan_df['disbursed_amount'].describe()
# Plot Boxplot to go in detail for outliers
chartil.uni_continuous_boxplot(loan_df, 'disbursed_amount')
chartil.uni_continuous_boxplot(train_test_combined_dataset, 'disbursed_amount') # More or less same distribution in combined dataset
# Treat Outlier at later stage

## A.3. asset_cost is a numeric 
loan_df['asset_cost'].describe() # looks like has some outlier 
# Plot Boxplot to go in detail for outliers
chartil.uni_continuous_boxplot(loan_df, 'asset_cost')
chartil.uni_continuous_boxplot(train_test_combined_dataset, 'asset_cost') # More or less same distribution in combined dataset
# Treat Outlier at later stage

## A.4. ltv is a numeric 
loan_df['ltv'].describe() # looks like has some outlier 
# Plot Boxplot to go in detail for outliers
chartil.uni_continuous_boxplot(loan_df, 'ltv')
chartil.uni_continuous_boxplot(train_test_combined_dataset, 'ltv') # More or less same distribution in combined dataset
# Treat Outlier at later stage

## A.5 branch_id - As per Data Dictionary it represents Branch where the loan was disbursed
# Looks like a Categorical variable with 82 unique values
chartil.uni_category_barchart(loan_df, 'branch_id',  limit_bars_count_to=20) # Majority of loan application are from branch 2, 67, 3 etc
# Convert to a categorical and check the bar chart
loan_df['branch_id'] = loan_df['branch_id'].astype('category')
train_test_combined_dataset['branch_id'] = train_test_combined_dataset['branch_id'].astype('category')  # Apply changes on loan_df to combined dataset

## A.6 supplier_id - As per Data Dictionary it represents Vehicle Dealer where the loan was disbursed. 
# Looks like Categorical with 2953 unique values
chartil.uni_category_barchart(loan_df, 'supplier_id', limit_bars_count_to=20) # Limited bars to 20 as there are 2953 unique values

# Convert to a categorical and check the bar chart
loan_df['supplier_id'] = loan_df['supplier_id'].astype('category')
train_test_combined_dataset['supplier_id'] = train_test_combined_dataset['supplier_id'].astype('category')  # Apply changes on loan_df to combined dataset
loan_df['supplier_id'].dtype

## A.7 manufacturer_id 
# Looks like a Categorical variable with 11 unique values
chartil.uni_category_barchart(loan_df, 'manufacturer_id') # Majority of loan application are for manufacturer_id 86 followed by 45, 51  branch 2, 67, 3 etc

# Convert to a categorical and check the bar chart
loan_df['manufacturer_id'] = loan_df['manufacturer_id'].astype('category')
train_test_combined_dataset['manufacturer_id'] = train_test_combined_dataset['manufacturer_id'].astype('category')  # Apply changes on loan_df to combined dataset
loan_df['manufacturer_id'].dtype

## A.8 Current_pincode_ID - Pincode of teh customer - Should not have any effect on default or not, lets check in subsequent steps
# Looks like categorical variable with 6698
chartil.uni_category_barchart(loan_df, 'Current_pincode_ID', limit_bars_count_to=25) # Majority of loan application comes from 2578, 1446 area
# Convert to a categorical and check the bar chart
loan_df['Current_pincode_ID'] = loan_df['Current_pincode_ID'].astype('category')
train_test_combined_dataset['Current_pincode_ID'] = train_test_combined_dataset['Current_pincode_ID'].astype('category')
loan_df['Current_pincode_ID'].dtype

## A.9 Date.of.Birth - Date of birth of the customer - A date filed in format dd-mm-yy
# Convert to datefield
loan_df['Date.of.Birth'] = pd.to_datetime(loan_df['Date.of.Birth'], format = "%d-%m-%y")
train_test_combined_dataset['Date.of.Birth'] = pd.to_datetime(train_test_combined_dataset['Date.of.Birth'], format = "%d-%m-%y")

# Unique values 15433
loan_df['Date.of.Birth'].describe() # Date varies from first 1969-01-01 00:00:00 Last 2068-12-31 00:00:00 - Unreasonable range
# There are some ambiquity with the date as it is only 2 digit, need to clean future dates with some threshold or cutoff
from datetime import timedelta, date
future = loan_df['Date.of.Birth'] > date(year=2018,month=10,day=31) # Max date as per DisbursalDate
loan_df.loc[future, 'Date.of.Birth'] -= timedelta(days=365.25*100)
# Apply same to combined dataset
future = train_test_combined_dataset['Date.of.Birth'] > date(year=2018,month=10,day=31) # Max date as per DisbursalDate
train_test_combined_dataset.loc[future, 'Date.of.Birth'] -= timedelta(days=365.25*100)

# Check the date distrinuction again
loan_df['Date.of.Birth'].describe() # Date varies from first 1949-09-15 Last 2000-10-20 - Range looks ok 
# Create additional column from this date - Mainly Age (Except Age, other features like birth month or date or year should not have effect on default - As a precaution derive those columns as well.
# Calculcate Age as on 31-10-2018 (Max date as per DisbursalDate)
loan_df['customer_age'] = ((pd.to_datetime('2018-10-31') - loan_df['Date.of.Birth']).dt.days)/365
loan_df['customer_age'] = loan_df['customer_age'].round()
loan_df['customer_age'].describe()

# Same apply to combined dataset
train_test_combined_dataset['customer_age'] = ((pd.to_datetime('2018-10-31') - train_test_combined_dataset['Date.of.Birth']).dt.days)/365
train_test_combined_dataset['customer_age'] = train_test_combined_dataset['customer_age'].round()
train_test_combined_dataset['customer_age'].describe()

# Add additional bin for age group
loan_df['customer_age_bin'] = pd.cut(loan_df['customer_age'], [0, 16, 19, 22, 30, 40, 50, 60, 70, 100], labels=['0-16', '16-19', '19-22', '22-30','30-40','40-50','50-60', '60-70','70-100'])
train_test_combined_dataset['customer_age_bin'] = pd.cut(train_test_combined_dataset['customer_age'], [0, 16, 19, 22, 30, 40, 50, 60, 70, 100], labels=['0-16', '16-19', '19-22', '22-30','30-40','40-50','50-60', '60-70','70-100'])

chartil.uni_category_barchart(loan_df, 'customer_age_bin', order_by_label=True)  
# Todo Analyse birth month, weekeday and other field has any effect on default
# Drop original Date.of.Birth column
loan_df = loan_df.drop("Date.of.Birth", axis=1)
train_test_combined_dataset = train_test_combined_dataset.drop("Date.of.Birth", axis=1)

## A.10 Employment.Type
# Looks like a Categorical variable with 3 unique values
chartil.uni_category_barchart(loan_df, 'Employment.Type')  # 7661 records is nan or unknown set it as new category value of Unknown (mostly unemployed or living on earlier earning/saving)
# Fill missing or Nan in Employment.Type with Unknown Category
loan_df['Employment.Type'].fillna('Unknown', inplace=True)
train_test_combined_dataset['Employment.Type'].fillna('Unknown', inplace=True)
# Convert to a categorical and check the bar chart
loan_df['Employment.Type'] = loan_df['Employment.Type'].astype('category')
train_test_combined_dataset['Employment.Type'] = train_test_combined_dataset['Employment.Type'].astype('category')
loan_df['Employment.Type'].dtype

## A.11 DisbursalDate, Date of disbursement
# Convert to datefield
loan_df['DisbursalDate'] = pd.to_datetime(loan_df['DisbursalDate'], format = "%d-%m-%y")
loan_df['DisbursalDate'].describe()  # 84 Unique values - Ranging from 01/08/2018 to 31/10/2018 - 3Month data
# Doesn't look like an important field as least possible imfluence on default - Also DisbursalDate date comes only after algo approves the new loan application. So cannot use for analysis
# May be max of this can be used to calculate DateOfBirth at the time of application. Otherwise it has of no use
# Drop the column
loan_df = loan_df.drop("DisbursalDate", axis=1)
train_test_combined_dataset = train_test_combined_dataset.drop("DisbursalDate", axis=1)

## A.12 State_ID
# Looks like a Categorical variable with 22 unique values
chartil.uni_category_barchart(loan_df, 'State_ID')  

# Convert to a categorical and check the bar chart
loan_df['State_ID'] = loan_df['State_ID'].astype('category')
train_test_combined_dataset['State_ID'] = train_test_combined_dataset['State_ID'].astype('category')

## A.13 Employee_code_ID - Employee of the organization who logged the disbursement
# Looks like a Categorical variable with 3270 unique values
chartil.uni_category_barchart(loan_df, 'Employee_code_ID', limit_bars_count_to=25)  

# Convert to a categorical
loan_df['Employee_code_ID'] = loan_df['Employee_code_ID'].astype('category')
train_test_combined_dataset['Employee_code_ID'] = train_test_combined_dataset['Employee_code_ID'].astype('category')
loan_df['Employee_code_ID'].dtype

## A.14 MobileNo_Avl_Flag
100*(loan_df['MobileNo_Avl_Flag'].astype('object').value_counts()/len(loan_df.index))
# Only one value available - Not usefull column, Need to drop 
loan_df = loan_df.drop("MobileNo_Avl_Flag", axis=1)
train_test_combined_dataset = train_test_combined_dataset.drop("MobileNo_Avl_Flag", axis=1)

## A.15 - Aadhar_flag - 2 unique values
100*(loan_df['Aadhar_flag'].astype('object').value_counts()/len(loan_df.index))
chartil.uni_category_barchart(loan_df, 'Aadhar_flag')  

## A.16 PAN_flag - 2 unique values
100*(loan_df['PAN_flag'].astype('object').value_counts()/len(loan_df.index))
chartil.uni_category_barchart(loan_df, 'PAN_flag')  

# A.17 - VoterID_flag - 2 unique values  
100*(loan_df['VoterID_flag'].astype('object').value_counts()/len(loan_df.index))
chartil.uni_category_barchart(loan_df, 'VoterID_flag')  

## A.18 - Driving_flag
100*(loan_df['Driving_flag'].astype('object').value_counts()/len(loan_df.index))
chartil.uni_category_barchart(loan_df, 'Driving_flag')  

## A.19 Passport_flag - 
100*(loan_df['Passport_flag'].astype('object').value_counts()/len(loan_df.index))
chartil.uni_category_barchart(loan_df, 'Passport_flag')  

## A.20 PERFORM_CNS.SCORE - Bureau Score - 573 unique values - Looks like contonous varbale 
loan_df['PERFORM_CNS.SCORE'].describe() # with range 0 to 890
chartil.uni_continuous_boxplot(loan_df, 'PERFORM_CNS.SCORE') # No outliers - Well balanced data 

## A.21 PERFORM_CNS.SCORE.DESCRIPTION - 20 unique values 
100*(loan_df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('object').value_counts()/len(loan_df.index))
# Basically contain groups like High Risj, Low Risk etc - Treat it as categorical variable
chartil.uni_category_barchart(loan_df, 'PERFORM_CNS.SCORE.DESCRIPTION')
# Convert to a categorical 
loan_df['PERFORM_CNS.SCORE.DESCRIPTION'] = loan_df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')
train_test_combined_dataset['PERFORM_CNS.SCORE.DESCRIPTION'] = train_test_combined_dataset['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')

## A.22 PRI.NO.OF.ACCTS - count of total loans taken by the customer at the time of disbursement - 108 Unique values 
loan_df['PRI.NO.OF.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.NO.OF.ACCTS') 
# Outliers detected, After say 10 account it doesn't look like have any chance (Treat at later stage)


## A.23 PRI.ACTIVE.ACCTS - Active loans at the time of disbursement - 40 Unique values - Ranging 0-144 - Skewed at top
loan_df['PRI.ACTIVE.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.ACTIVE.ACCTS') # Outliers detected, treat at later stage


## A.24 PRI.OVERDUE.ACCTS - Default account at the time of disbursement - 22 unique values ranging from 0-25
loan_df['PRI.OVERDUE.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.OVERDUE.ACCTS')
# Will deal weith outliers later (If required - since range is narrow)

## A.25 PRI.CURRENT.BALANCE - Total Principal outstanding at the time of disbursement - 71341 unique values 
loan_df['PRI.CURRENT.BALANCE'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.CURRENT.BALANCE') 
# Some are negative and some are positive - Outstanding principal must be in single sign
# Lets convert all negative to positive
loan_df['PRI.CURRENT.BALANCE'] = loan_df['PRI.CURRENT.BALANCE'].abs()
train_test_combined_dataset['PRI.CURRENT.BALANCE'] = train_test_combined_dataset['PRI.CURRENT.BALANCE'].abs()
# Will deal weith outliers later (If required)


## A.26 - PRI.SANCTIONED.AMOUNT - total amount that was sanctioned for all the loans at the time of disbursement - 44390 unqiue values
loan_df['PRI.SANCTIONED.AMOUNT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.SANCTIONED.AMOUNT') 
# Will deal weith outliers later (If required - since range is narrow)

## A.27 - PRI.DISBURSED.AMOUNT - total amount that was disbursed for all the loans at the time of disbursement - 47909 unique values ranging from 0 - 1.000000e+09 
loan_df['PRI.DISBURSED.AMOUNT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRI.DISBURSED.AMOUNT') 
# Will deal weith outliers later (If required - since range is narrow)

## A.28 SEC.NO.OF.ACCTS - count of total loans taken by the customer at the time of disbursement - 37 Unique values (Co applicant)
loan_df['SEC.NO.OF.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.NO.OF.ACCTS') 
# Outliers detected, After say 10 account it doesn't look like have any chance
# Will deal weith outliers later (If required - since range is narrow)

## A.29 SEC.ACTIVE.ACCTS - Active loans at the time of disbursement - 23 Unique values - Ranging 0-36
loan_df['SEC.ACTIVE.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.ACTIVE.ACCTS') # Outliers detected
# Will deal with outliers later

## A.30 SEC.OVERDUE.ACCTS - Default account at the time of disbursement - 9 unique values ranging from 0-8
loan_df['SEC.OVERDUE.ACCTS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.OVERDUE.ACCTS')
# Will deal weith outliers later (If required - since range is narrow)

## A.31 SEC.CURRENT.BALANCE - Total Principal outstanding at the time of disbursement - 3246 unique values 
loan_df['SEC.CURRENT.BALANCE'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.CURRENT.BALANCE') 
# Some are negative and some are positive - Outstanding principal must be in single sign
# Lets convert all negative to positive
loan_df['SEC.CURRENT.BALANCE'] = loan_df['SEC.CURRENT.BALANCE'].abs()
train_test_combined_dataset['SEC.CURRENT.BALANCE'] = train_test_combined_dataset['SEC.CURRENT.BALANCE'].abs()
# Will deal weith outliers later (If required - since range is narrow)

## A.32 - SEC.SANCTIONED.AMOUNT - total amount that was sanctioned for all the loans at the time of disbursement - 2223 unqiue values
loan_df['SEC.SANCTIONED.AMOUNT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.SANCTIONED.AMOUNT') 
# Will deal weith outliers later (If required - since range is narrow)

## A.33 - SEC.DISBURSED.AMOUNT - total amount that was disbursed for all the loans at the time of disbursement - 2553 unique values ranging from 0 -3.000000e+07
loan_df['SEC.DISBURSED.AMOUNT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.DISBURSED.AMOUNT') 
# Will deal weith outliers later (If required )

## A.34 - PRIMARY.INSTAL.AMT - EMI Amount of the primary loan - Ranging b/w 0-2.564281e+07
loan_df['PRIMARY.INSTAL.AMT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'PRIMARY.INSTAL.AMT') 
# Will deal weith outliers later (If required)

## A.35 - SEC.INSTAL.AMT - EMI Amount of the secondary loan - 1918 unique values ranging b/w 0-4.170901e+06
loan_df['SEC.INSTAL.AMT'].describe()
chartil.uni_continuous_boxplot(loan_df, 'SEC.INSTAL.AMT') 
# Will deal weith outliers later (If required - since range is narrow)

## A.36 - NEW.ACCTS.IN.LAST.SIX.MONTHS - 26 unique values rnging from 0-35
loan_df['NEW.ACCTS.IN.LAST.SIX.MONTHS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'NEW.ACCTS.IN.LAST.SIX.MONTHS')
# Will deal with outliers later (If required - since range is narrow)

## A.37 - DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS - 14 unique values ranging from 0-20
loan_df['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].describe()
chartil.uni_continuous_boxplot(loan_df, 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS')
# Will deal with outliers later (If required - since range is narrow)

## A.38 - AVERAGE.ACCT.AGE - The data is in teh form of "Xyrs xmon" - 192 unique values ranging from ???????????????????????????????????????????????????????????????????????
loan_df['AVERAGE.ACCT.AGE'].describe()
loan_df['AVERAGE.ACCT.AGE'] = loan_df['AVERAGE.ACCT.AGE'].apply(parseYrsMonStr)
train_test_combined_dataset['AVERAGE.ACCT.AGE'] = train_test_combined_dataset['AVERAGE.ACCT.AGE'].apply(parseYrsMonStr)
chartil.uni_continuous_boxplot(loan_df, 'AVERAGE.ACCT.AGE')
# Will deal with outliers later (If required - since range is narrow)

## A.39 -  CREDIT.HISTORY.LENGTH': 294
loan_df['CREDIT.HISTORY.LENGTH'].describe()  
loan_df['CREDIT.HISTORY.LENGTH'] = loan_df['CREDIT.HISTORY.LENGTH'].apply(parseYrsMonStr)
train_test_combined_dataset['CREDIT.HISTORY.LENGTH'] = train_test_combined_dataset['CREDIT.HISTORY.LENGTH'].apply(parseYrsMonStr)
chartil.uni_continuous_boxplot(loan_df, 'CREDIT.HISTORY.LENGTH')
# Will deal with outliers later (If required - since range is narrow)


## A.40 - NO.OF_INQUIRIES': 25 unique values range 0-36
loan_df['NO.OF_INQUIRIES'].describe()  
chartil.uni_continuous_boxplot(loan_df, 'NO.OF_INQUIRIES')
# Will deal with outliers later (If required - since range is narrow)

# (B) EDA - Analyse Defaulter pattern among various features


# Next round of cleanup
# Additional features from primary and scondary account
train_test_combined_dataset['HYPOTHETICAL.NEW.LOAN.EMI'] = train_test_combined_dataset['disbursed_amount']/36 # Assuming 3 year loan

train_test_combined_dataset['TOTAL.NO.OF.ACCTS'] = train_test_combined_dataset['PRI.NO.OF.ACCTS'] + train_test_combined_dataset['SEC.NO.OF.ACCTS']
train_test_combined_dataset['TOTAL.ACTIVE.ACCTS'] = train_test_combined_dataset['PRI.ACTIVE.ACCTS'] + train_test_combined_dataset['SEC.ACTIVE.ACCTS']
train_test_combined_dataset['TOTAL.OVERDUE.ACCTS'] = train_test_combined_dataset['PRI.OVERDUE.ACCTS'] + train_test_combined_dataset['SEC.OVERDUE.ACCTS']
train_test_combined_dataset['TOTAL.CURRENT.BALANCE'] = train_test_combined_dataset['PRI.CURRENT.BALANCE'] + train_test_combined_dataset['SEC.CURRENT.BALANCE']
train_test_combined_dataset['TOTAL.SANCTIONED.AMOUNT'] = train_test_combined_dataset['PRI.SANCTIONED.AMOUNT'] + train_test_combined_dataset['SEC.SANCTIONED.AMOUNT']
train_test_combined_dataset['TOTAL.DISBURSED.AMOUNT'] = train_test_combined_dataset['PRI.DISBURSED.AMOUNT'] + train_test_combined_dataset['SEC.DISBURSED.AMOUNT']
train_test_combined_dataset['TOTAL.INSTAL.AMT'] = train_test_combined_dataset['PRIMARY.INSTAL.AMT'] + train_test_combined_dataset['SEC.INSTAL.AMT']
train_test_combined_dataset['IMMEDIATE.OVERDUE.EMI'] = (train_test_combined_dataset['TOTAL.INSTAL.AMT']*train_test_combined_dataset['TOTAL.OVERDUE.ACCTS'])/train_test_combined_dataset['TOTAL.ACTIVE.ACCTS']
train_test_combined_dataset['AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'] = train_test_combined_dataset['TOTAL.CURRENT.BALANCE']/(12*train_test_combined_dataset['AVERAGE.ACCT.AGE'])
train_test_combined_dataset['AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'] = train_test_combined_dataset['TOTAL.NO.OF.ACCTS']/train_test_combined_dataset['CREDIT.HISTORY.LENGTH']

train_test_combined_dataset['IMMEDIATE.OVERDUE.EMI'].fillna(0, inplace=True)
train_test_combined_dataset['IMMEDIATE.OVERDUE.EMI'].replace(np.inf,0,inplace=True)
train_test_combined_dataset['AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'].fillna(0, inplace=True)
train_test_combined_dataset['AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'].replace(np.inf,0,inplace=True)
train_test_combined_dataset['AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'].fillna(0, inplace=True)
train_test_combined_dataset['AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'].replace(np.inf,0,inplace=True)

# Do it for Primry account
train_test_combined_dataset['PRI.IMMEDIATE.OVERDUE.EMI'] = (train_test_combined_dataset['PRIMARY.INSTAL.AMT']*train_test_combined_dataset['PRI.OVERDUE.ACCTS'])/train_test_combined_dataset['PRI.ACTIVE.ACCTS']
train_test_combined_dataset['PRI.AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'] = train_test_combined_dataset['PRI.CURRENT.BALANCE']/(12*train_test_combined_dataset['AVERAGE.ACCT.AGE'])
train_test_combined_dataset['PRI.AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'] = train_test_combined_dataset['PRI.NO.OF.ACCTS']/train_test_combined_dataset['CREDIT.HISTORY.LENGTH']
 
train_test_combined_dataset['PRI.IMMEDIATE.OVERDUE.EMI'].fillna(0, inplace=True)
train_test_combined_dataset['PRI.IMMEDIATE.OVERDUE.EMI'].replace(np.inf,0,inplace=True)
train_test_combined_dataset['PRI.AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'].fillna(0, inplace=True)
train_test_combined_dataset['PRI.AVG.MONTHLY.EMI.TOPAY.ONALL.LOANS'].replace(np.inf,0,inplace=True)
train_test_combined_dataset['PRI.AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'].fillna(0, inplace=True)
train_test_combined_dataset['PRI.AVG.LOAN.ACCOUNT.OPENED.PER.YEAR'].replace(np.inf,0,inplace=True)

train_test_combined_dataset['OVERDUE.BURDEN.ABOVE.NEW.EMI'] = train_test_combined_dataset['PRI.IMMEDIATE.OVERDUE.EMI']>=train_test_combined_dataset['HYPOTHETICAL.NEW.LOAN.EMI']
train_test_combined_dataset['OVERDUE.BURDEN.ABOVE.NEW.EMI'] = train_test_combined_dataset['OVERDUE.BURDEN.ABOVE.NEW.EMI'].astype('category')


train_test_combined_dataset.isnull().sum() 



# (C) Model Building

# Final transformation for categorical variable

train_test_combined_dataset['loan_default'] = train_test_combined_dataset['loan_default'].astype('int')

# select all categorical variables
df_categorical = train_test_combined_dataset.select_dtypes(include=['object', 'category'])
df_categorical = df_categorical.drop("dataset_type", axis=1) # Exclude dataset_type from LabelEncoder transformation

df_categorical.head()


# apply Label encoder to df_categorical
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df_categorical.head()

# concat df_categorical with original df
train_test_combined_dataset = train_test_combined_dataset.drop(df_categorical.columns, axis=1)
train_test_combined_dataset = pd.concat([train_test_combined_dataset, df_categorical], axis=1)

train_test_combined_dataset.head()

# convert target variable loan_default to categorical
train_test_combined_dataset['loan_default'] = train_test_combined_dataset['loan_default'].astype('category')

# Smote - Balancing
defaulter_set_in_training = train_test_combined_dataset[train_test_combined_dataset['loan_default']==1]
defaulter_set_in_training = defaulter_set_in_training[defaulter_set_in_training['dataset_type']=='train']

train_test_combined_dataset = pd.concat([defaulter_set_in_training, train_test_combined_dataset])

# Separate train and external test based on dummy column created at the begining
source_training_dataset = train_test_combined_dataset[(train_test_combined_dataset['dataset_type']=='train')]
source_external_test_dataset = train_test_combined_dataset[(train_test_combined_dataset['dataset_type']=='test')]

# Drop the column dataset_type which was introduced to separate train and test in combined dataset
source_training_dataset = source_training_dataset.drop("dataset_type", axis=1)
source_external_test_dataset = source_external_test_dataset.drop("dataset_type", axis=1)

source_training_dataset.info()

chartil.bi_continuous_category_category_crosstab_percentage(loan_df, 'PERFORM_CNS.SCORE.DESCRIPTION', 'loan_default')

# Split internal train and test dataset
# Importing train-test-split 
from sklearn.model_selection import train_test_split

X = source_training_dataset.drop(['loan_default','UniqueID'], axis=1)
y = source_training_dataset['loan_default'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# Import required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# GridSearchCV to find optimal n_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# For XGboost
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance




#####################     Decision Tree     #####################

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train, y_train)

# Let's check the evaluation metrics of our default model

# Making predictions
y_pred_default = dt_default.predict(X_test)

# Printing classification report
print(classification_report(y_test, y_pred_default))

# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))



#####################     Random Forest     #####################
# Default Hyperparameter

# Running the random forest with default parameters.
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

# Making predictions
predictions = rfc.predict(X_test)

# Let's check the report of our default model
print(classification_report(y_test,predictions))

# Printing confusion matrix
print(confusion_matrix(y_test,predictions))

print(accuracy_score(y_test,predictions))

# Fro ROC curve you need probability 
y_pred = rfc.predict_proba(X_test)
# evaluate predictions
roc = metrics.roc_auc_score(y_test, y_pred[:, 1])
print("AUC: %.2f%%" % (roc * 100.0))

# Hyperparameter Tuning



# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

# instantiate the model
rf = RandomForestClassifier()
    
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="accuracy")
rf.fit(X_train, y_train)

# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()

# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




# Next check number of trees parameter (n_estimators)



# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'n_estimators': range(25, 500, 25)}

# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=7)


# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="accuracy")
rf.fit(X_train, y_train)

# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()

# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [6,7,8],
    'min_samples_leaf': range(100, 500, 100),
    'min_samples_split': range(100, 500, 100),
    'n_estimators': range(10, 111, 20), 
    'max_features': [6, 8, 10, 12]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1,verbose = 1, scoring="roc_auc")

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)

# We can get accuracy of 0.7828463239934562 using {'max_depth': 8, 'max_features': 12, 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 10}
# RF Final model using optimised params
rfOptimal = RandomForestClassifier(bootstrap=True,
                             max_depth=8,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=12,
                             n_estimators=10)

rfOptimal.fit(X_train,y_train)
predictions = rfOptimal.predict(X_test)

# Fro ROC curve you need probability 
y_pred = rfOptimal.predict_proba(X_test)
# evaluate predictions
roc = metrics.roc_auc_score(y_test, y_pred[:, 1])
print("AUC: %.2f%%" % (roc * 100.0))
# AUC: 62.87%










## XGBoost
# fit model on training data with default hyperparameters


model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
# use predict_proba since we need probabilities to compute auc
y_pred = model.predict_proba(X_test)
y_pred[:10]

# evaluate predictions
roc = metrics.roc_auc_score(y_test, y_pred[:, 1])
print("AUC: %.2f%%" % (roc * 100.0))
# 65.11%


def xgbParamTuner(params, cvFolds=2) :
    xgb_model = XGBClassifier(n_estimators=200)
    
    # set up GridSearchCV()
    model_cv = GridSearchCV(estimator = xgb_model, 
                    param_grid = params, 
                    scoring= 'roc_auc', 
                    cv = cvFolds, 
                    verbose = 1,
                    return_train_score=True,
                    n_jobs = -1)     

    # fit the model
    model_cv.fit(X_train, y_train) 
    
    # scores of GridSearch CV
    scores = model_cv.cv_results_
    return pd.DataFrame(scores)
      
    # printing the optimal accuracy score and hyperparameters
    print('We can get accuracy of',model_cv.best_score_,'using',model_cv.best_params_)
    
#xgbParamTuner(params = {'max_depth': [3,4], 'subsample': [0.3, 0.6]}, cvFolds=2)

gridScores = xgbParamTuner(params={'max_depth': [2,3,4,5],'learning_rate': [0.1, 0.2, 0.3, 0.6], 'subsample': [0.3, 0.6, 0.9]}, cvFolds=3)    


# Previous best roc reult We can get accuracy of 0.6377871418797078 using {'learning_rate': 0.2, 'subsample': 0.9}


# Final model with XGboost
params = {'learning_rate': 0.1,
          'max_depth': 4, 
          'n_estimators':250,
          'subsample':0.9
         } # 'objective':'binary:logistic' outputs probability rather than label, which we need for auc

# fit model on training data
xbgOptimal = XGBClassifier(params = params)
xbgOptimal.fit(X_train, y_train)
print(xbgOptimal.score(X_train, y_train))

# predict
y_pred = xbgOptimal.predict_proba(X_test)
y_pred[:10]

y_pred_clf = xbgOptimal.predict(X_test)

print(confusion_matrix(y_test,y_pred_clf))
# roc_auc
auc = metrics.roc_auc_score(y_test, y_pred[:, 1])
auc

# Next CatBoost
# Todo


# Submission prediction
finalModel = xbgOptimal
predictions = xbgOptimal.predict(data=source_external_test_dataset.drop(['UniqueID', 'loan_default'], axis=1))
sum(predictions)
submission_df = pd.concat([source_external_test_dataset.drop(['loan_default'], axis=1), pd.DataFrame(predictions, columns=['loan_default'])], axis=1)

submission_df.to_csv("D:\\projects\\PythonWS\\LTFS\\submission.csv", columns=['UniqueID','loan_default'], index=False)



