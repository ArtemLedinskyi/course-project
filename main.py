import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score

# Зчитуємо дані
white_wine_data = pd.read_csv("winequality-white.csv", delimiter=";")
red_wine_data = pd.read_csv("winequality-red.csv", delimiter=";")

# Об'єднуємо дані
wine_data = pd.concat([white_wine_data, red_wine_data])

# Попередня обробка даних
wine_data.dropna(inplace=True)
wine_data["quality"] = wine_data["quality"].astype(int)

# Візуалізація дані
sns.pairplot(wine_data, vars=wine_data.columns[:-1], hue="quality")
plt.show()

# Розділення на навчальний та тестувальний набори
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормалізація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Вибір моделей та крос-валідація
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy: {scores.mean():.4f}, Standard Deviation: {scores.std():.4f}")

# Навчання та оцінка найкращої моделі
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model (Random Forest) Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, zero_division='warn'))
