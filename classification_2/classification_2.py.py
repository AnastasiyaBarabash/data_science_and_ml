import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score

file_path = 'Volgmed_2013.xlsx'
df = pd.read_excel(file_path)

required_columns = ['Пол', 'Окружность грудной клетки в покое, см', 'Экскурсия грудной клетки, см', 'Челночный бег, с']
df = df[required_columns]
df = df.dropna()

for col in ['Окружность грудной клетки в покое, см', 'Экскурсия грудной клетки, см', 'Челночный бег, с']:
    df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
    df[col] = pd.to_numeric(df[col])

df['Пол'] = df['Пол'].apply(lambda x: 1 if x.lower() == 'жен' else 0)

X = df[['Окружность грудной клетки в покое, см', 'Экскурсия грудной клетки, см', 'Челночный бег, с']]
y = df['Пол']

if X.empty:
    raise ValueError("Нет данных после очистки")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\nЧисло истинно отрицательных (TN): {tn}")
print(f"Число ложно положительных (FP): {fp}")
print(f"Число ложно отрицательных (FN): {fn}")
print(f"Число истинно положительных (TP): {tp}")

incorrect_idx = (y_pred != y_test).values

X_test_original = X_test * scaler.scale_ + scaler.mean_

X_test_df = pd.DataFrame(X_test_original, columns=['Окружность грудной клетки в покое, см', 'Экскурсия грудной клетки, см', 'Челночный бег, с'])
X_test_df['Пол'] = y_test.reset_index(drop=True)
X_test_df['Предсказанный пол'] = y_pred

features = ['Окружность грудной клетки в покое, см', 'Экскурсия грудной клетки, см', 'Челночный бег, с']
colors = ['blue', 'red']

plt.figure(figsize=(18, 5))

for i, (feat1, feat2) in enumerate([(0, 1), (0, 2), (1, 2)]):
    plt.subplot(1, 3, i + 1)

    correct = ~incorrect_idx
    plt.scatter(
        X_test_df.loc[correct, features[feat1]],
        X_test_df.loc[correct, features[feat2]],
        c=X_test_df.loc[correct, 'Пол'].apply(lambda x: colors[x]),
        marker='o',
        label='Верно классифицированные',
        edgecolor='k',
        alpha=0.7
    )

    plt.scatter(
        X_test_df.loc[incorrect_idx, features[feat1]],
        X_test_df.loc[incorrect_idx, features[feat2]],
        c=X_test_df.loc[incorrect_idx, 'Пол'].apply(lambda x: 'orange' if x == 1 else 'green'),
        marker='x',
        s=100,
        label='Неверно классифицированные'
    )

    plt.xlabel(features[feat1])
    plt.ylabel(features[feat2])
    plt.title(f'Диаграмма рассеяния: {features[feat1]} vs {features[feat2]}')
    plt.legend()

plt.tight_layout()
plt.show()

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Сбалансированная точность (Balanced Accuracy): {balanced_acc:.2f}")

f1 = f1_score(y_test, y_pred)
print(f"F1-мера: {f1:.2f}")

diagnostic_odds_ratio = (tp * tn) / (fp * fn)
print(f"коэффициент диагностической вероятности (DOR): {diagnostic_odds_ratio:.2f}")