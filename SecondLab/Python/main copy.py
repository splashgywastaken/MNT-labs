import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# Главный класс приложения
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Classifier")

        # Инициализация переменных
        self.model = None
        self.scaler = None
        self.encoder = None

        # Элементы интерфейса
        frame_left = tk.Frame(root)
        frame_left.pack(side=tk.LEFT, padx=10, pady=10)

        frame_right = tk.Frame(root)
        frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

        # Элементы интерфейса слева
        self.label = tk.Label(frame_left, text="Загрузите набор данных")
        self.label.pack()

        self.load_button = tk.Button(frame_left, text="Загрузить CSV", command=self.load_data)
        self.load_button.pack()

        self.train_button = tk.Button(frame_left, text="Обучить модель", command=self.train_model)
        self.train_button.pack()

        self.progress = ttk.Progressbar(frame_left, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack()

        self.result_label = tk.Label(frame_left, text="Результаты:")
        self.result_label.pack()

        self.check_button = tk.Button(frame_left, text="Выполнить предсказание", command=self.check_model)
        self.check_button.pack()

        self.data = None
        self.tree = None

        # Элементы интерфейса справа (таблица)
        self.create_table(frame_right)

    # Функция для создания таблицы
    def create_table(self, parent):
        # Создаем таблицу
        columns = ("Feature1", "Feature2", "Feature3", "Outcome")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings")

        # Заголовки столбцов
        for col in columns:
            self.tree.heading(col, text=col)

        # Определяем ширину каждого столбца
        self.tree.column("Feature1", width=100)
        self.tree.column("Feature2", width=100)
        self.tree.column("Feature3", width=100)
        self.tree.column("Outcome", width=100)

        self.tree.pack()

        # Кнопка для добавления строки в таблицу
        add_row_button = tk.Button(parent, text="Добавить строку", command=self.add_row)
        add_row_button.pack()

    # Функция для добавления новой строки
    def add_row(self):
        self.tree.insert('', 'end', values=("Введите", "значения", "сюда", ""))

    # Функция для загрузки данных
    def load_data(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filepath:
            try:
                self.data = pd.read_csv(filepath)
                self.label.config(text=f"Загружен датасет: {filepath.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить CSV файл: {e}")

    # Функция для обучения модели
    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Ошибка", "Загрузите данные перед обучением!")
            return

        # Разделение данных на признаки и целевую переменную
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        # Преобразование категориальных меток в числовые
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)

        # Масштабирование данных
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Создание нейронной сети
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(len(np.unique(y)), activation='softmax'))

        # Компиляция модели с использованием градиентного спуска и метода моментов
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)  # метод моментов
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Обучение модели
        self.progress['maximum'] = 100
        self.progress['value'] = 0
        self.root.update_idletasks()

        history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)

        self.progress['value'] = 100
        self.root.update_idletasks()

        # Оценка модели
        y_pred = np.argmax(self.model.predict(X_test), axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        self.result_label.config(text=f"Точность модели: {accuracy:.2f}")

    # Функция для проверки модели на новых данных
    def check_model(self):
        if self.model is None or self.scaler is None or self.encoder is None:
            messagebox.showwarning("Ошибка", "Сначала обучите модель!")
            return

        # Получаем данные из таблицы
        table_data = []
        for row in self.tree.get_children():
            values = self.tree.item(row)["values"]
            table_data.append([float(values[0]), float(values[1]), float(values[2])])  # Преобразуем введенные значения

        # Преобразуем данные в массив numpy и масштабируем
        X_check = np.array(table_data)
        X_check = self.scaler.transform(X_check)

        # Прогнозирование
        predictions = np.argmax(self.model.predict(X_check), axis=-1)
        predicted_labels = self.encoder.inverse_transform(predictions)

        # Заполняем столбец Outcome предсказанными значениями
        for row, pred in zip(self.tree.get_children(), predicted_labels):
            self.tree.set(row, column="Outcome", value=pred)


# Запуск программы
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
