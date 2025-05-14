import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Load raw data
df = pd.read_csv('heart.csv')

#Seperate features and target variables
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

#Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(exclude='object').columns.tolist()

#Set up column transformer
preprocessor = ColumnTransformer([('num', StandardScaler(), numerical_cols), 
                                   ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])     

# Transform the data
X_processed = preprocessor.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

#Save processed data
pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test).to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Preprocessing complete. Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
