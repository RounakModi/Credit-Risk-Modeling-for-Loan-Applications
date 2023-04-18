# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Set random seed for reproducibility
np.random.seed(42)

# Create dummy data
n_samples = 10000
loan_amounts = np.random.normal(5000, 1000, n_samples)
loan_terms = np.random.choice([12, 24, 36, 48, 60], n_samples)
interest_rates = np.random.normal(0.1, 0.02, n_samples)
credit_scores = np.random.normal(700, 100, n_samples)
employment_statuses = np.random.choice(['Employed', 'Unemployed', 'Self-employed'], n_samples)
default_probabilities = np.random.beta(5, 5, n_samples)
defaults = np.random.binomial(1, default_probabilities)

# Create a pandas DataFrame from the dummy data
loan_data = pd.DataFrame({
    'loan_amount': loan_amounts,
    'loan_term': loan_terms,
    'interest_rate': interest_rates,
    'credit_score': credit_scores,
    'employment_status': employment_statuses,
    'default_probability': default_probabilities,
    'default': defaults
})

# Perform exploratory data analysis
sns.pairplot(loan_data[['loan_amount', 'loan_term', 'interest_rate', 'credit_score', 'default']], hue='default')

# Preprocess the data
X = loan_data.drop(['default_probability', 'default'], axis=1)
y = loan_data['default']
le = LabelEncoder()
X['employment_status'] = le.fit_transform(X['employment_status'])
ohe = OneHotEncoder(sparse=False)
ohe_df = pd.DataFrame(ohe.fit_transform(X[['employment_status']]))
X = pd.concat([X, ohe_df], axis=1)
X = X.drop('employment_status', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(), params, cv=5)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]

# Evaluate the model
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))
print('ROC AUC score:', roc_auc_score(y_test, y_prob))
