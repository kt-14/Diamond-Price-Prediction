import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import numpy as np

# Load data from the CSV file
data = pd.read_csv('diamonds.csv')

# Define features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Preprocess categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['cut', 'color', 'clarity'])
    ],
    remainder='passthrough'
)

# Create a pipeline
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = make_pipeline(preprocessor, model)

# Train the model
pipeline.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Extract feature names after preprocessing
preprocessor.fit(X)
categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['cut', 'color', 'clarity'])
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
feature_names = np.concatenate([categorical_features, numerical_features])

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Exclude all features related to clarity and 'carat'
importance_df = importance_df[~importance_df['Feature'].str.contains('clarity|carat')]

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Get the top 2 important features excluding clarity and carat
top_features = importance_df.head(4)

print("Top 4 Important Features:")
print(top_features)
