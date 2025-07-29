import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
imputer=SimpleImputer(strategy='median')
df=pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
print(df.isnull().sum().value_counts())
df['target'] = diabetes.target

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and parameter grids
models = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'Support Vector Regressor': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=42),
        'params': {
            'alpha': [0.1, 1, 10],
            'l1_ratio': [0.5, 0.7, 0.9]
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    }
}

# Perform Grid Search
results = []

for model_name, mp in models.items():
    print(f"Training {model_name}...")
    clf = GridSearchCV(mp['model'], 
                      mp['params'], 
                      cv=5, 
                      scoring='r2',
                      n_jobs=-1)
    
    clf.fit(X_train_scaled, y_train)
    
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    model_results = {
        'Model': model_name,
        'Best Parameters': clf.best_params_,
        'R² Score': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
    
    results.append(model_results)

# Create results DataFrame
results_df = pd.DataFrame(results).sort_values(by='R² Score', ascending=False)
print("\nModel Comparison Results:")
print(results_df)

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(x='R² Score', y='Model', data=results_df, palette='viridis')
plt.title('Model Performance Comparison (R² Score)')
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.show()

# Feature Importance for best model
best_model_name = results_df.iloc[0]['Model']
best_model_params = results_df.iloc[0]['Best Parameters']

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print(f"\nFeature Importance for {best_model_name}:")
    model = models[best_model_name]['model'].set_params(**best_model_params)
    model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.Series(model.feature_importances_, 
                                  index=diabetes.feature_names)
    feature_importance.sort_values(ascending=False).plot(kind='barh')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.show()
