import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Įkeliam duomenis
with open("src/Xy_data.pkl", "rb") as f:
    X, y = pickle.load(f)

print(f"Duomenų kiekis: {len(X)}, Požymių skaičius: {len(X[0])}")

# Dalinam į train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Sukuriam ir apmokam modelį
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Prognozė
y_pred = model.predict(X_test)

# Tikslumas
acc = accuracy_score(y_test, y_pred)
print(f"✅ Tikslumas: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Vizualizacija
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Wake', 'Stage 1'], yticklabels=['Wake', 'Stage 1'])
plt.xlabel('Prognozuota')
plt.ylabel('Tikroji')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Papildoma info
print("\nIšsami klasifikavimo ataskaita:")
print(classification_report(y_test, y_pred, target_names=['Wake', 'Stage 1']))


import pickle
import os

# Sukuriam katalogą, jei reikia
os.makedirs("models", exist_ok=True)

# Išsaugom modelį
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelis išsaugotas į models/svm_model.pkl")