import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Шаг 1: Чтение данных
file_path = 'Volgmed_2013.xlsx'
df = pd.read_excel(file_path)

# Фильтруем данные для мужчин
round = df[df['Пол'] == 'муж']['Окружность грудной клетки в покое, см']
ves = df[df['Пол'] == 'муж']['Вес, кг']

# Объединяем данные в DataFrame для удобной работы
data = pd.DataFrame({'Окружность': round, 'Вес': ves})

# Убираем пропуски
data = data.dropna()

# Очистка от выбросов методом межквартильного размаха
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Фильтруем выбросы
data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Упорядочиваем по весу
data_cleaned = data_cleaned.sort_values(by='Вес')

# Шаг 2: Линейная регрессия
X = data_cleaned['Вес'].values.reshape(-1, 1)  # Признак
y = data_cleaned['Окружность'].values  # Целевая переменная

# Добавляем константу к X для расчета интерсепта
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))

# Модель линейной регрессии
model = LinearRegression()
model.fit(X_with_intercept, y)

# Коэффициенты регрессии
intercept = model.intercept_
slope = model.coef_[1]  # Наклон
theta = np.mean(X) - np.mean(y)

# Предсказания
y_hat = model.predict(X_with_intercept)

# Коэффициент детерминации (R²)
r_squared = model.score(X_with_intercept, y)

# Вычисление p-значений для коэффициентов
n = len(X)
p_value_intercept = 2 * (1 - stats.t.cdf(np.abs(intercept / (np.std(y) / np.sqrt(n))), df=n-2))
p_value_slope = 2 * (1 - stats.t.cdf(np.abs(slope / (np.std(X) / np.sqrt(n))), df=n-2))

# Шаг 3: Визуализация
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Вес', y='Окружность', data=data_cleaned, color='blue', label='Данные')
plt.plot(data_cleaned['Вес'], y_hat, color='red', label=f'Линия регрессии')
plt.xlabel('Вес (кг)')
plt.ylabel('Окружность грудной клетки в покое (см)')
plt.title('Диаграмма рассеяния и линия регрессии')
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print(f'Коэффициент наклона: {slope:.3f}')
print(f'Свободный член: {intercept:.3f}')
print(f'Коэффициент детерминации (R²): {r_squared:.3f}')
print(f'p-значение для наклона: {p_value_slope:.5f}')
print(f'p-значение для свободного члена: {p_value_intercept:.5f}')

# Шаг 3: Стохастические моделирования для оценки p-value и R²
bootstrap_slopes = []
bootstrap_intercepts = []
bootstrap_r_squared = []
iterations = 20000

for _ in range(iterations):
    indices = np.random.choice(range(len(X)), size=len(X), replace=True)
    X_sample = X[indices]
    y_sample = y[indices]

    # Добавляем константу к X для бутстрап-модели
    X_sample_with_intercept = np.hstack((np.ones((X_sample.shape[0], 1)), X_sample))
    
    # Обучаем модель на бутстрап-выборке
    model.fit(X_sample_with_intercept, y_sample)
    
    # Сохраняем наклон и интерсепт
    bootstrap_slopes.append(model.coef_[1])
    bootstrap_intercepts.append(model.intercept_)
    
    # Предсказания для бутстрап-выборки
    y_hat_sample = model.predict(X_sample_with_intercept)

    # Расчет R² для бутстрап-выборки
    ss_res = np.sum((y_sample - y_hat_sample) ** 2)
    ss_tot = np.sum((y_sample - np.mean(y_sample)) ** 2)
    r_squared_bootstrap = 1 - (ss_res / ss_tot)
    bootstrap_r_squared.append(r_squared_bootstrap)

# Оценка p-value на основе bootstrap
bootstrap_slopes = np.array(bootstrap_slopes)
bootstrap_intercepts = np.array(bootstrap_intercepts)
p_value_slope_bootstrap = np.mean(np.abs(bootstrap_slopes) >= np.abs(slope))
p_value_intercept_bootstrap = np.mean(np.abs(bootstrap_intercepts) >= np.abs(intercept))

# Средний R² на основе бутстрапа
mean_r_squared_bootstrap = np.mean(bootstrap_r_squared)

print(f'p-значение для наклона на основе моделирования: {p_value_slope_bootstrap:.5f}')
print(f'p-значение для свободного члена на основе моделирования: {p_value_intercept_bootstrap:.5f}')
print(f'Средний коэффициент детерминации (R²) на основе моделирования: {mean_r_squared_bootstrap:.3f}')