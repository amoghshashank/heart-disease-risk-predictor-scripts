import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Load the processed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Initialize the model
model = LogisticRegression(max_iter=1000)

#Optional: Grid search for tuning
param_grid = {'C': [0.1, 1, 10],}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)
model = grid.best_estimator_

#Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Save model
joblib.dump(model, 'logistic_model.joblib')
print("Model saved as logistic_model.joblib")