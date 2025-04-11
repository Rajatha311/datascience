
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv(r"Titanic-Dataset.csv")
test_df = pd.read_csv(r"Titanic-Dataset.csv")


def preprocess_data(df):
 
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    df['Age'].fillna(df['Age'].median(), inplace=True)

    if df['Embarked'].isnull().sum() > 0:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    label = LabelEncoder()
    df['Sex'] = label.fit_transform(df['Sex'])  
    df['Embarked'] = label.fit_transform(df['Embarked'])  
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

X = train_df.drop(['Survived', 'PassengerId'], axis=1)
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))


sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X_test = test_df.drop(columns=['PassengerId'], errors='ignore')

if 'Survived' in X_test.columns:
    X_test = X_test.drop(columns=['Survived'])


test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})


submission.to_csv('titanic_predictions.csv', index=False)
print("Submission file saved as 'titanic_predictions.csv'")
