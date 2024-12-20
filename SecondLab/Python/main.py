# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import zscore

# Reload the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)

# Replace zero values with NaN in specified columns
columns_to_nullify = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
data[columns_to_nullify] = data[columns_to_nullify].replace(0, np.nan)

# Handle missing values by imputing with median
imputer = SimpleImputer(strategy='median')
data[columns_to_nullify] = imputer.fit_transform(data[columns_to_nullify])

# Detect and handle outliers using Z-score (for specific columns)
zscore_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for col in zscore_columns:
    z_scores = zscore(data[col])
    threshold = 3  # Define Z-score threshold
    max_val = data[col][z_scores <= threshold].max()
    data[col] = np.where(z_scores > threshold, max_val, data[col])

# Detect and handle outliers using IQR (for specific columns)
iqr_columns = ['Pregnancies', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
for col in iqr_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    max_val = data[col][data[col] <= upper_bound].max()
    data[col] = np.where(data[col] > upper_bound, max_val, data[col])

# Normalize the data using Min-Max Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.iloc[:, :-1])  # Exclude the target column
X = pd.DataFrame(scaled_data, columns=data.columns[:-1])
y = data.iloc[:, -1]  # Target variable

# Balance the dataset by oversampling the minority class
class_counts = y.value_counts()
majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

# Oversampling: Duplicate samples from the minority class
minority_samples = X[y == minority_class]
minority_labels = y[y == minority_class]
oversampled_X = pd.concat([X, minority_samples.sample(class_counts[majority_class], replace=True)])
oversampled_y = pd.concat([y, minority_labels.sample(class_counts[majority_class], replace=True)])

# Train-test split (85% training, 15% test)
X_train, X_test, y_train, y_test = train_test_split(oversampled_X, oversampled_y, test_size=0.15, random_state=42, stratify=oversampled_y)

# Train MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)
mlp_train_accuracy = accuracy_score(y_train, mlp_model.predict(X_train))
mlp_test_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))

# Train RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# Output results
mlp_train_accuracy, mlp_test_accuracy, rf_train_accuracy, rf_test_accuracy
