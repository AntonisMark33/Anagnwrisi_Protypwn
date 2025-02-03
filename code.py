import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sklearn


def load_data(file_path: str):
    """Load housing dataset."""
    return pd.read_csv(file_path)


def preprocess_data(numeric_attributes, categorical_attributes):
    """Create preprocessing pipelines for numerical and categorical data."""
    # Numerical data preprocessing
    numerical_transform = Pipeline(steps=[
        ('missing_imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Standardize the data
    ])

    # Categorical data preprocessing
    categorical_transform = Pipeline(steps=[
        ('missing_imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categories
        ('encoder', OneHotEncoder())  # One-hot encode categorical features
    ])

    # Combine both numerical and categorical transformers
    data_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numerical_transform, numeric_attributes),
            ('categorical', categorical_transform, categorical_attributes)
        ])
    return data_transformer


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate the model."""
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Evaluation metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    return train_mse, test_mse, train_mae, test_mae


def cross_validate_model(model, features, target, cv_splitter):
    """Perform cross-validation and return mean MSE."""
    cv_scores = cross_val_score(model, features, target, cv=cv_splitter, scoring='neg_mean_squared_error')
    return -cv_scores.mean()


# 1. Load dataset
data_frame = load_data("housing.csv")

# 2. Preprocessing configuration
numeric_attributes = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
                      'households', 'median_income']
categorical_attributes = ['ocean_proximity']
data_transformer = preprocess_data(numeric_attributes, categorical_attributes)

# 3. Split dataset into features (X) and target (y)
features = data_frame[numeric_attributes + categorical_attributes]
target = data_frame['median_house_value']

# 4. Data Visualization
data_frame.hist(bins=50, figsize=(20, 15))
plt.show()

plt.figure(figsize=(10, 8))
sns.pairplot(data_frame[['longitude', 'latitude', 'median_house_value']])
plt.show()

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 5.1 Perceptron Model
perceptron_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('classifier', Perceptron())
])

perceptron_results = train_and_evaluate_model(perceptron_model, X_train, y_train, X_test, y_test)
print("Περεστροφή - Εκπαίδευση MSE:", perceptron_results[0])
print("Περεστροφή - Δοκιμή MSE:", perceptron_results[1])
print("Περεστροφή - Εκπαίδευση MAE:", perceptron_results[2])
print("Περεστροφή - Δοκιμή MAE:", perceptron_results[3])

# 5.2 Linear Regression Model
linear_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('regressor', LinearRegression())
])

linear_results = train_and_evaluate_model(linear_model, X_train, y_train, X_test, y_test)
print("Γραμμική Παλινδρόμηση - Εκπαίδευση MSE:", linear_results[0])
print("Γραμμική Παλινδρόμηση - Δοκιμή MSE:", linear_results[1])
print("Γραμμική Παλινδρόμηση - Εκπαίδευση MAE:", linear_results[2])
print("Γραμμική Παλινδρόμηση - Δοκιμή MAE:", linear_results[3])

# 5.3 MLP Regressor Model
mlp_model = Pipeline(steps=[
    ('data_preprocessing', data_transformer),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
])

mlp_results = train_and_evaluate_model(mlp_model, X_train, y_train, X_test, y_test)
print("MLP - Εκπαίδευση MSE:", mlp_results[0])
print("MLP - Δοκιμή MSE:", mlp_results[1])
print("MLP - Εκπαίδευση MAE:", mlp_results[2])
print("MLP - Δοκιμή MAE:", mlp_results[3])

# 6. Cross-Validation
cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)

# Perceptron CV
perceptron_cv_mean_mse = cross_validate_model(perceptron_model, features, target, cv_splitter)
print("Περεστροφή - Μέσος Όρος CV MSE:", perceptron_cv_mean_mse)

# Linear Regression CV
linear_cv_mean_mse = cross_validate_model(linear_model, features, target, cv_splitter)
print("Γραμμική Παλινδρόμηση - Μέσος Όρος CV MSE:", linear_cv_mean_mse)

# MLP CV
mlp_cv_mean_mse = cross_validate_model(mlp_model, features, target, cv_splitter)
print("MLP - Μέσος Όρος CV MSE:", mlp_cv_mean_mse)