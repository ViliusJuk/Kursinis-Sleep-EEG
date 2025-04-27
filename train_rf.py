import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Įkeliam duomenis
with open("src/Xy_data.pkl", "rb") as f:
    X, y = pickle.load(f)

# Dalinam į train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apmokam Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Tikslumas
acc = accuracy_score(y_test, y_pred)
print(f"🌲 Random Forest tikslumas: {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', linewidths=1.5, linecolor='black',
            xticklabels=['Wake', 'Stage 1'], yticklabels=['Wake', 'Stage 1'],
            annot_kws={"size": 14})
plt.xlabel('Prognozuota', fontsize=12)
plt.ylabel('Tikroji', fontsize=12)
plt.title(f'Random Forest Confusion Matrix (Tikslumas: {acc:.2f})', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/confusion_matrix_rf.png")
plt.show()

# Ataskaita
report = classification_report(y_test, y_pred, target_names=['Wake', 'Stage 1'])
print("🌲 Random Forest klasifikavimo ataskaita:\n")
print(report)

with open("reports/classification_report_rf.txt", "w") as f:
    f.write(report)

    # Užtikrinam, kad katalogas egzistuoja
os.makedirs("reports", exist_ok=True)

# Išsaugom klasifikacijos ataskaitą į .txt failą
with open("reports/classification_report_rf.txt", "w") as f:
    f.write(report)

# Išsaugom modelį
os.makedirs("models", exist_ok=True)
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Random Forest modelis išsaugotas į models/random_forest_model.pkl")
print("✅ Random Forest klasifikacijos ataskaita išsaugota į reports/classification_report_rf.txt")
