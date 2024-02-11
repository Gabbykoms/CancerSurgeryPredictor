import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the CSV file into a DataFrame
df = pd.read_csv('BRCA.csv')

# Check for missing values
df['Patient_Status'] = df['Patient_Status'].map({'Alive': 0, 'Dead': 1})
print(df.isnull().sum())

# Impute missing values in numeric columns with the mean
numeric_columns = ['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4']
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Impute missing values in categorical columns with the mode
categorical_columns = ['Gender', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']
for column in categorical_columns:
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)


    

# Drop Date_of_Last_Visit column if it's not essential
df.drop(columns=['Date_of_Last_Visit'], inplace=True)
df.drop(columns=['Patient_ID', 'Date_of_Surgery'], inplace=True)



# Drop rows with missing values in Patient_Status column
df.dropna(subset=['Patient_Status'], inplace=True)

# Recheck for missing values
print(df.isnull().sum())

# Plot histograms for numeric features
numeric_columns = ['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4']
df[numeric_columns].hist(bins=20, figsize=(12, 8))
plt.show()

# Plot box plots for numeric features
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numeric_columns])
plt.show()

sns.pairplot(df[numeric_columns])
plt.show()

# Plot count plot for the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Patient_Status', data=df)
plt.show()


# Plot pie chart for the target variable
plt.figure(figsize=(6, 6))
df['Patient_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.show()

# Step 2: Preprocess Non-Numeric Columns
# Convert 'Gender' to binary numeric values
df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})


# Perform one-hot encoding for categorical variables
categorical_cols = ['Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Step 3: Select Relevant Features
relevant_features = df_encoded.columns.tolist()  # Include all encoded columns
# Exclude 'Patient_Status' if it's the target variable

if 'Patient_Status' in relevant_features:
    df_subset = df_encoded[relevant_features]  # Create df_subset with all relevant features
    relevant_features.remove('Patient_Status')

# Step 4: Compute the Correlation Matrix
corr_matrix = df_subset.corr()

# Step 5: Visualize the Correlation Matrix as a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10}, linewidths=0.5, linecolor='black', square=True)
plt.title('Correlation Matrix of Relevant Features')
plt.show()



# Split the dataset into features (X) and target variable (y)
X = df_subset.drop(columns=['Patient_Status'])
y = df_subset['Patient_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(rf_classifier, 'model.pkl')

