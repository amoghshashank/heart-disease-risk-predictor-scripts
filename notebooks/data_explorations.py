import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#Load dataset
df = pd.read_csv('heart.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows of the dataset:\n", df.head())

#Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

#Class balance plot
sns.set(style="whitegrid")
sns.countplot(x='HeartDisease', data=df)
plt.title('Heart Disease presence (0 = No, 1 = Yes)')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_balance.png')
plt.show()