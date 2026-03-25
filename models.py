import StandardScaler


def models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("KNeighborsClassifier модель")
   
    knn_classifier = KNeighborsClassifier()

    # Обучаем модель на масштабированных данных
    knn_classifier.fit(X_train_scaled, y_train)

    # Делаем предсказание
    y_train_pred_knn = knn_classifier.predict(X_train_scaled)
    y_test_pred_knn = knn_classifier.predict(X_test_scaled)

    # Оцениваем точность модели
    accuracy_knn_train = accuracy_score(y_train, y_train_pred_knn)

    print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_knn_train:.4f}")

    # Тестовая выборка
    accuracy_knn_test = accuracy_score(y_test, y_test_pred_knn)

    print("\nТЕСТОВАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_knn_test:.4f}")
    #________________________________________
    print("DecisionTreeClassifier модель")
    dt_classifier = DecisionTreeClassifier(max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5, random_state=42)

    # Обучение модели на тренировочных данных
    dt_classifier.fit(X_train, y_train)

    # Получение предсказаний на тестовых данных
    y_train_pred_dt = dt_classifier.predict(X_train)
    y_test_pred_dt = dt_classifier.predict(X_test)

    # Оценка точности модели
    accuracy_dt_train = accuracy_score(y_train, y_train_pred_dt)

    print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_dt_train:.4f}")
    # Тестовая выборка
    accuracy_dt_test = accuracy_score(y_test, y_test_pred_dt)
    print("\nТЕСТОВАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_dt_test:.4f}")

    print("logisticRegression модель")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    y_train_pred_lr = model.predict(X_train_scaled)
    y_test_pred_lr = model.predict(X_test_scaled)
    accuracy_lr_train = accuracy_score(y_train, y_train_pred_lr)

    print("ОБУЧАЮЩАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_lr_train:.4f}")
    accuracy_lr_test = accuracy_score(y_test, y_test_pred_lr)
    print("\nТЕСТОВАЯ ВЫБОРКА:")
    print(f"  Accuracy:  {accuracy_lr_test:.4f}")
