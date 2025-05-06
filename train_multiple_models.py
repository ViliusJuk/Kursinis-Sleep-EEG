import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Įkeliam duomenis
with open("src/Xy_data.pkl", "rb") as f:
    X, y = pickle.load(f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modeliai
models = {
    'SVM (linear)': SVC(kernel='linear'),
    'SVM (rbf)': SVC(kernel='rbf'),
    'SVM (poly)': SVC(kernel='poly'),
    'SVM (sigmoid)': SVC(kernel='sigmoid'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Katalogai
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

results = []

# Modelių treniravimas ir analizė
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Wake', 'Stage 1'])
    cm = confusion_matrix(y_test, y_pred)

    # Išsaugom modelį
    model_filename = f"models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    # Išsaugom ataskaitą
    report_path = f"reports/{name.replace(' ', '_').replace('(', '').replace(')', '')}_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion matrix paveikslėlis
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Wake', 'Stage 1'], yticklabels=['Wake', 'Stage 1'],
                linewidths=1.5, linecolor='black', annot_kws={"size": 14})
    plt.xlabel('Prognozuota', fontsize=12)
    plt.ylabel('Tikroji', fontsize=12)
    plt.title(f'{name} (Tikslumas: {acc:.2f})', fontsize=14)
    plt.tight_layout()
    cm_path = f"reports/{name.replace(' ', '_').replace('(', '').replace(')', '')}_confusion.png"
    plt.savefig(cm_path)
    plt.close()

    results.append({
        "Modelis": name,
        "Tikslumas": round(acc, 2),
        "Report_path": report_path,
        "CM_image": cm_path,
        "Model_pkl": model_filename
    })

# Rezultatų lentelė
results_df = pd.DataFrame(results)
results_df.to_csv("reports/modeliai_rezultatai.csv", index=False)
print("✅ Visi modeliai apmokyti ir rezultatai išsaugoti į reports/")