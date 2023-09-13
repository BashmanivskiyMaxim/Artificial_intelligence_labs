import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=column_names, sep=",\s*", engine="python")

# Вибираємо ознаки та цільову змінну
features = data.drop("income", axis=1)
target = (data["income"] == ">50K").astype(int)

# Перекодування категоріальних ознак
categorical_features = features.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    features[feature] = label_encoders[feature].fit_transform(features[feature])

# Масштабування числових ознак
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Розбиття даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Навчання моделі SVM
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)

# Прогнозування на тестовому наборі
y_pred = svm_classifier.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
