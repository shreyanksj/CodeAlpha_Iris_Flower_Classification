import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('Iris.csv')

# Display the first 5 rows
print(df.head())

# Display basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Visualize the count of each species
sns.countplot(x='Species', data=df)
plt.title('Count of Each Iris Species')
plt.show()

# Pairplot to see relationships between features
sns.pairplot(df, hue='Species')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(['Id', 'Species'], axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Drop the 'Id' column
df = df.drop('Id', axis=1)

# Separate features and target
X = df.drop('Species', axis=1)
y = df['Species']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'iris_logreg_model.pkl')
loaded_model = joblib.load('iris_logreg_model.pkl')

sample = [[5.1, 3.5, 1.4, 0.2]]  # Example measurements
predicted_species = loaded_model.predict(sample)
print("Predicted species for sample {}: {}".format(sample, predicted_species[0]))