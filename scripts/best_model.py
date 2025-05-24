import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

#Load data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

#Tune model
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

#Final model
best_model = grid.best_estimator_
joblib.dump(best_model, 'best_model.joblib')

print("X_test shape:", X_test.shape)
print(X_test.head())

#Evaluate
y_pred = best_model.predict(X_test)
print("Best Random Forest Model")
print(classification_report(y_test, y_pred))