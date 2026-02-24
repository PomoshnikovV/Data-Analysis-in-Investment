# Модель: y = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ + ε
#
# Где:
#   y    — целевая переменная (то, что предсказываем)
#   x    — признаки (features)
#   β    — коэффициенты (веса), которые модель учит
#   ε    — ошибка
#
# Метрики качества:
#   MAE  — средняя абсолютная ошибка
#   MSE  — среднеквадратичная ошибка
#   RMSE — корень из MSE
#   R²   — коэффициент детерминации (доля объяснённой дисперсии)
#
# Ключевые концепции:
#   - Train/Test split: разделение данных на обучающую и тестовую выборки
#   - Переобучение (overfitting): модель "заучивает" данные, плохо обобщает
#   - Feature engineering: создание информативных признаков
# ===========================================================================


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)

sber = df[df['Ticker'] == 'SBER'].copy()

X = sber[['Open']].values
y = sber['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nКоэффициент (β₁): {model.coef_[0]:.4f}")
print(f"Свободный член (β₀): {model.intercept_:.4f}")
print(f"\nМетрики на тестовой выборке:")
print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R²:   {r2_score(y_test, y_pred):.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test, y_test, alpha=0.5, s=15, label='Реальные данные')

x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
ax.plot(x_line, model.predict(x_line), color='red', linewidth=2, label='Регрессия')
ax.set_title('Линейная регрессия: Open → Close (SBER)', fontsize=14)
ax.set_xlabel('Цена открытия')
ax.set_ylabel('Цена закрытия')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_3_1_simple_regression.png', dpi=150)
plt.close()
print("График сохранён -> plot_3_1_simple_regression.png")