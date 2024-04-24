import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('Dataset_spine.csv')
data = df.drop(columns=['Unnamed: 13']) 

data.dropna(inplace=True)

label_encoder = LabelEncoder()
data['Class_att'] = label_encoder.fit_transform(data['Class_att'])

X = data.drop(columns=['Class_att'])
y = data['Class_att']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

new_data = pd.DataFrame({
    'Col1': [65.2, 42.8, 68.7, 50.3],
    'Col2': [23.5, 18.9, 27.4, 15.6],
    'Col3': [38.7, 34.2, 41.8, 29.5],
    'Col4': [30.1, 27.3, 35.6, 22.8],
    'Col5': [112.4, 108.9, 116.3, 105.2],
    'Col6': [4.8, 6.2, 5.4, 3.9],
    'Col7': [0.1, 0.2, -0.3, 0.5],
    'Col8': [13.9, 15.2, 14.6, 12.3],
    'Col9': [17.2, 19.8, 16.5, 18.1],
    'Col10': [11.9, 10.5, 12.8, 9.7],
    'Col11': [-23.4, -27.8, -20.5, -25.9],
    'Col12': [21.5, 19.6, 23.1, 18.7]
})

new_predictions = rf_classifier.predict(new_data)

predicted_classes = label_encoder.inverse_transform(new_predictions)

print("Predicted Class Labels:")
print(predicted_classes)