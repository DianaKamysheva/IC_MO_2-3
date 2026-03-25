import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import warnings

warnings.filterwarnings('ignore')


def main():
    print("=" * 50)
    print("ПОДГОТОВКА ДАННЫХ К ОБУЧЕНИЮ")
    print("=" * 50)

    # 1. Загрузка данных
    df = pd.read_csv('telecom_churn.csv')
    print(f"\nЗагружено данных: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # 2. Отделение целевой переменной
    print("\n1. Отделение целевой переменной:")
    print("-" * 30)

    # Проверяем наличие колонки churn
    if 'churn' in df.columns:
        X = df.drop('churn', axis=1)
        y = df['churn'].astype(int)  # Преобразуем bool в int (0/1)
        print(f"Признаки X: {X.shape[1]} столбцов")
        print(f"Целевая переменная y: {y.shape[0]} значений")
        print(f"Распределение: 0={sum(y == 0)}, 1={sum(y == 1)}")
        print(f"Доля оттока: {y.mean():.2%}")
    else:
        print("Ошибка: колонка 'churn' не найдена в датасете!")
        print(f"Доступные колонки: {df.columns.tolist()}")
        return

    # 3. Разделение на обучающую и тестовую выборки
    print("\n2. Разделение данных на train/test (75%/25%):")
    print("-" * 30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Обучающая выборка: {X_train.shape[0]} строк")
    print(f"Тестовая выборка: {X_test.shape[0]} строк")
    print(f"Random state = 42")
    print(f"Stratify = True (сохранение пропорции классов)")
    print(f"\nРаспределение в train: 0={sum(y_train == 0)}, 1={sum(y_train == 1)} ({y_train.mean():.2%})")
    print(f"Распределение в test: 0={sum(y_test == 0)}, 1={sum(y_test == 1)} ({y_test.mean():.2%})")

    # 4. Определение типов признаков
    print("\n3. Определение числовых и категориальных признаков:")
    print("-" * 30)

    # Исключаем колонки, которые не будем использовать как признаки
    exclude_cols = ['phone number']  # phone number - это идентификатор

    # Определяем числовые признаки
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Определяем категориальные признаки (включая строки и bool)
    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Удаляем исключенные колонки
    numerical_features = [col for col in numerical_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]

    print(f"Числовые признаки ({len(numerical_features)}): {numerical_features}")
    print(f"Категориальные признаки ({len(categorical_features)}): {categorical_features}")

    if 'phone number' in X.columns:
        print(f"Исключен признак: 'phone number' (идентификатор)")

    # Проверяем, что все колонки учтены
    all_features = numerical_features + categorical_features
    if len(all_features) != X.shape[1]:
        print(f"Внимание: не все колонки учтены! Всего колонок: {X.shape[1]}, учтено: {len(all_features)}")
        missing_cols = set(X.columns) - set(all_features) - set(exclude_cols)
        if missing_cols:
            print(f"Неучтенные колонки: {missing_cols}")

    # 5. Создание препроцессора
    print("\n4. Создание препроцессора:")
    print("-" * 30)
    print("Для числовых признаков: StandardScaler")
    print("Для категориальных признаков: OneHotEncoder (handle_unknown='ignore', drop='first')")

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )

    # 6. Применение препроцессора
    print("\n5. Применение препроцессора к данным:")
    print("-" * 30)

    try:
        # Обучаем препроцессор на обучающей выборке
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        print(f"Размер обработанных данных (train): {X_train_processed.shape}")
        print(f"Размер обработанных данных (test): {X_test_processed.shape}")

        # Проверяем на NaN
        if np.isnan(X_train_processed).any():
            print("Внимание: обнаружены NaN значения в обработанных данных!")
            print(f"Количество NaN: {np.isnan(X_train_processed).sum()}")

    except Exception as e:
        print(f"Ошибка при применении препроцессора: {e}")
        return

    # 7. Сохранение результатов
    print("\n6. Сохранение результатов:")
    print("-" * 30)

    # Сохраняем препроцессор
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✓ Препроцессор сохранен в 'preprocessor.pkl'")

    # Сохраняем выборки
    np.save('X_train.npy', X_train_processed)
    np.save('X_test.npy', X_test_processed)
    np.save('y_train.npy', y_train.values)
    np.save('y_test.npy', y_test.values)
    print("✓ Обработанные данные сохранены в .npy файлы")

    # Сохраняем имена признаков
    feature_names = numerical_features.copy()
    for cat_feature in categorical_features:
        try:
            encoder = preprocessor.named_transformers_['cat']
            cat_features_names = encoder.get_feature_names_out([cat_feature])
            # Убираем первый уровень (drop='first')
            if len(cat_features_names) > 1:
                feature_names.extend(cat_features_names[1:])
            else:
                feature_names.extend(cat_features_names)
        except:
            print(f"Предупреждение: не удалось получить имена признаков для {cat_feature}")

    with open('feature_names.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_names))
    print(f"✓ Имена признаков сохранены в 'feature_names.txt' ({len(feature_names)} признаков)")

    # Сохраняем дополнительную информацию
    info = {
        'n_features': X_train_processed.shape[1],
        'n_train_samples': len(y_train),
        'n_test_samples': len(y_test),
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'excluded_features': exclude_cols,
        'train_distribution': {'0': int(sum(y_train == 0)), '1': int(sum(y_train == 1))},
        'test_distribution': {'0': int(sum(y_test == 0)), '1': int(sum(y_test == 1))}
    }
    joblib.dump(info, 'data_info.pkl')
    print("✓ Информация о данных сохранена в 'data_info.pkl'")

    print("\n" + "=" * 50)
    print("Подготовка данных завершена")
    print("=" * 50)

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()