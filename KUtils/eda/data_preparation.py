df.head(10)
df.info()
df.describe()
df.shape

df['mainroad'] = df['mainroad'].map({'yes':1, 'no':0})