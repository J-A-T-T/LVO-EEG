import numpy as np
import pandas as pd

df = pd.read_csv('./data/MAESTRO_DATA.csv')

print(df.head(5))

# Remove row which we are not supposed to use
df = df[df['Notes'] != 'DO NOT USE. No consent']


# time elapsed
df['reached_hospital'] = pd.to_datetime(df['reached_hospital'])
df['eeg_d_t'] = pd.to_datetime(df['eeg_d_t'])
df['time_elapsed'] = df['eeg_d_t'] -df['reached_hospital']
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'])
df['time_elapsed'] = df['time_elapsed'].dt.total_seconds().div(60).astype(int)

# Count missing values for each column
print(df.isnull().sum())

#Create dummy variables
df['Male'] = [1 if s == 1 else 0 for s in df['gender']]
df['Female'] = [1 if s == 2 else 0 for s in df['gender']]

df.to_csv('MAESTRO_DATA_CLEAN.csv', index=False)

# gauge correlation between columns
print(df.corr(method ='pearson')['LVO (incl. M2)'])
df['stroke'].replace(2,1,inplace=True)
df['stroke'].replace(3,1,inplace=True)
df['stroke'].replace(4,1,inplace=True)
df['stroke'].replace(5,1,inplace=True)
df_subset = df[['age', 'Male', 'Female', 'lams', 'nihss', 'LVO (incl. M2)','time_elapsed','stroke']]
df_subset = pd.DataFrame(df_subset)


df_subset.fillna(0, inplace=True)
df_subset.to_csv('./data/df_onsite.csv')


