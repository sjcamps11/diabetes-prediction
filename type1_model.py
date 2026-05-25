import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

os.makedirs('Prototype/Model', exist_ok=True)

csv_pth = "Data/cleaned_diabetes.csv"
df = pd.read_csv(csv_pth)

print(df.columns.tolist())

featureColumns =[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


targetColumn = "Outcome"

x = df[featureColumns]
y = df[targetColumn]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2,
    random_state=42,
    stratify=y
    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

Kval = range(1, 21)
accyScore = []

for k in Kval:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    preditcions = knn.predict(X_test_scaled)
    accy = accuracy_score(y_test, preditcions)

    accyScore.append(accy)

bestk = list(Kval)[accyScore.index(max(accyScore))]


print(f"best k value: {bestk}")

finalmodel = KNeighborsClassifier(n_neighbors=bestk)
finalmodel.fit(X_train_scaled, y_train)

finalPredict = finalmodel.predict(X_test_scaled)
finalAccy = accuracy_score(y_test, finalPredict)

print(f"accuracy: {finalAccy: .2%}")

model_data = {
    "model": finalmodel,
    "scaler": scaler,
    "features": featureColumns,
    "target": targetColumn,
    "accuracy": finalAccy
    }

with open("Prototype/Model/diabetes_knn_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Model saved successfully.")
