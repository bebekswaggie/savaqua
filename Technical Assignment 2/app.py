import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load data
data = pd.read_csv('ai4i2020.csv')

# Cek jika ada nilai yang kosong/hilang
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# hapus kolom yang datanya kosong/hilang
data = data.dropna()

# visualisai data
plt.figure(figsize=(12, 6))
sns.histplot(data['Tool wear [min]'], kde=True)
plt.title('Tool Wear Distribution')
plt.show()

# Visualisi korelasi matrix
plt.figure(figsize=(12, 6))
numerical_data = data.select_dtypes(include=['float', 'int'])
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# mendefinisikan variabel x dan y
# menghapus variabel yang tidak diperlukan untuk variabel x
X = data.drop(columns=['Machine failure', 'UDI', 'Product ID'])
# variabel y
y = data['Machine failure']

# mengidentifikasi dan menghilangkan variabel yang bukan numerik
non_numeric_cols = X.select_dtypes(exclude=['float', 'int']).columns
print("Non-numeric columns:", non_numeric_cols)
X = X.drop(columns=non_numeric_cols)

# membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#  Penskalaan fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Pemilihan model Gradient Boosting Classifier
model_gb = GradientBoostingClassifier(random_state=42)
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(estimator=model_gb, param_grid=param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
best_model_gb = grid_search_gb.best_estimator_
print("Best parameters for Gradient Boosting: ", grid_search_gb.best_params_)

# Evaluasi
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, roc_auc

metrics_gb = evaluate_model(best_model_gb, X_test, y_test)

print(f'Gradient Boosting - Accuracy: {metrics_gb[0]:.4f}, Precision: {metrics_gb[1]:.4f}, Recall: {metrics_gb[2]:.4f}, F1-score: {metrics_gb[3]:.4f}, ROC AUC: {metrics_gb[4]:.4f}')

# Confusion matrix untuk Gradient Boosting
conf_matrix_gb = confusion_matrix(y_test, best_model_gb.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# grafik probabilitas kegagalan berdasarkan "Rotational speed [rpm]" untuk setiap kegagalan
for failure_type in y_test.unique():
    df_failure = data[data['Machine failure'] == failure_type]
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df_failure, x='Rotational speed [rpm]')
    plt.title(f'Probability of Failure with respect to Rotational speed [rpm] ({failure_type})')
    plt.ylabel('Probability Density')
    plt.show()
