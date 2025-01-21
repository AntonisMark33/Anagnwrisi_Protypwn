import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load dataset
data_frame = pd.read_csv("housing.csv")

# 2. Data Preprocessing

# Identifying numerical and categorical features
numeric_attributes = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
categorical_attributes = ['ocean_proximity']

# Handling missing values: fill numerical attributes with median values
numerical_transform = Pipeline(steps=[
    ('missing_imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Encoding categorical attributes
categorical_transform = Pipeline(steps=[
    ('missing_imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Combine transformers into a ColumnTransformer
data_transformer = ColumnTransformer(
    transformers=[
        ('numeric', numerical_transform, numeric_attributes),
        ('categorical', categorical_transform, categorical_attributes)
    ])

# 3. Split dataset into features (X) and target (y)
features = data_frame[numeric_attributes + categorical_attributes]
target = data_frame['median_house_value']

# 4. Data Visualization (Histograms and Scatter Plots)
data_frame.hist(bins=50, figsize=(20, 15))
plt.show()

sns.pairplot(data_frame[['longitude', 'latitude', 'median_house_value']])
plt.show()

# 5. Model Training and Evaluation

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 5.1 Training with Perceptron (Linear Classification)
perceptron_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('classifier', Perceptron())
])

# Train the model
perceptron_model.fit(X_train, y_train)

# Predictions and error calculation
train_predictions_perceptron = perceptron_model.predict(X_train)
test_predictions_perceptron = perceptron_model.predict(X_test)

# Evaluation
print("Perceptron - Train MSE:", mean_squared_error(y_train, train_predictions_perceptron))
print("Perceptron - Test MSE:", mean_squared_error(y_test, test_predictions_perceptron))
print("Perceptron - Train MAE:", mean_absolute_error(y_train, train_predictions_perceptron))
print("Perceptron - Test MAE:", mean_absolute_error(y_test, test_predictions_perceptron))

# 5.2 Training with Least Squares Algorithm (Linear Regression)
least_squares_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('regressor', LinearRegression())
])

# Train the model
least_squares_model.fit(X_train, y_train)

# Predictions and error calculation
train_predictions_ls = least_squares_model.predict(X_train)
test_predictions_ls = least_squares_model.predict(X_test)

# Evaluation
print("Least Squares - Train MSE:", mean_squared_error(y_train, train_predictions_ls))
print("Least Squares - Test MSE:", mean_squared_error(y_test, test_predictions_ls))
print("Least Squares - Train MAE:", mean_absolute_error(y_train, train_predictions_ls))
print("Least Squares - Test MAE:", mean_absolute_error(y_test, test_predictions_ls))

# 5.3 Training with Multi-layer Perceptron (Neural Network)
mlp_regressor_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
])

# Train the model
mlp_regressor_model.fit(X_train, y_train)

# Predictions and error calculation
train_predictions_mlp = mlp_regressor_model.predict(X_train)
test_predictions_mlp = mlp_regressor_model.predict(X_test)

# Evaluation
print("MLP - Train MSE:", mean_squared_error(y_train, train_predictions_mlp))
print("MLP - Test MSE:", mean_squared_error(y_test, test_predictions_mlp))
print("MLP - Train MAE:", mean_absolute_error(y_train, train_predictions_mlp))
print("MLP - Test MAE:", mean_absolute_error(y_test, test_predictions_mlp))

# 6. 10-fold Cross-Validation
cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)

# Perceptron
perceptron_cv_scores = cross_val_score(perceptron_model, features, target, cv=cv_splitter, scoring='neg_mean_squared_error')
print("Perceptron - CV Mean MSE:", -perceptron_cv_scores.mean())

# Least Squares
least_squares_cv_scores = cross_val_score(least_squares_model, features, target, cv=cv_splitter, scoring='neg_mean_squared_error')
print("Least Squares - CV Mean MSE:", -least_squares_cv_scores.mean())

# MLP
mlp_cv_scores = cross_val_score(mlp_regressor_model, features, target, cv=cv_splitter, scoring='neg_mean_squared_error')
print("MLP - CV Mean MSE:", -mlp_cv_scores.mean())