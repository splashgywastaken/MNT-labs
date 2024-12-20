import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import csv
import random

# Размер холста
CANVAS_SIZE = 32
PIXEL_SIZE = 20  # Размер каждого пикселя
LEARNING_RATE = 0.1


# Нейрон с одним слоем
class SimpleNeuron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [0.5] * input_size  # Инициализируем веса значением 0.5
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = sum(x * w for x, w in zip(inputs, self.weights))
        return 1 if summation >= 0 else 0

    def train(self, inputs, expected_output):
        prediction = self.predict(inputs)
        error = expected_output - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]

    def get_weights(self):
        return self.weights

    def set_weights(self, new_weights):
        self.weights = new_weights

    def randomize_weights(self):
        self.weights = [random.uniform(-1, 1) for _ in range(len(self.weights))]


# Главный класс приложения
class SymbolRecognizerApp:
    def __init__(self, rootArg):
        self.action_label = None
        self.randomize_button = None
        self.import_button = None
        self.export_button = None
        self.clear_button = None
        self.no_button = None
        self.yes_button = None
        self.learning_rate_entry = None
        self.learning_rate_label = None
        self.recognized_symbol_text = None
        self.recognized_symbol_label = None
        self.pixel_rects = None
        self.canvas = None
        self.root = rootArg
        self.root.title("Symbol Recognizer")

        # Инициализация нейрона
        self.neuron = SimpleNeuron(CANVAS_SIZE * CANVAS_SIZE, LEARNING_RATE)

        # Инициализация изображения и рисования
        self.canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)  # Белый фон
        self.draw = ImageDraw.Draw(self.canvas_image)

        # Создание интерфейса
        self.create_interface()

    def create_interface(self):
        # Холст для рисования
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE * PIXEL_SIZE, height=CANVAS_SIZE * PIXEL_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Графическое отображение данных пикселей (прямоугольники вместо размытого изображения)
        self.pixel_rects = [
            [
                self.canvas.create_rectangle(
                    i * PIXEL_SIZE, j * PIXEL_SIZE,
                    (i + 1) * PIXEL_SIZE, (j + 1) * PIXEL_SIZE,
                    fill="white", outline="gray"
                ) for j in range(CANVAS_SIZE)
            ] for i in range(CANVAS_SIZE)
        ]

        # Поле для распознанного символа
        self.recognized_symbol_label = tk.Label(self.root, text="Распознанный символ:")
        self.recognized_symbol_label.grid(row=1, column=0)

        self.recognized_symbol_text = tk.Entry(self.root, width=10)
        self.recognized_symbol_text.grid(row=1, column=1)
        self.recognized_symbol_text.insert(0, "?")

        # Поле для ввода веса обучения
        self.learning_rate_label = tk.Label(self.root, text="Вес обучения:")
        self.learning_rate_label.grid(row=2, column=0)

        self.learning_rate_entry = tk.Entry(self.root, width=10)
        self.learning_rate_entry.grid(row=2, column=1)
        self.learning_rate_entry.insert(0, str(LEARNING_RATE))

        # Кнопки YES и NO
        self.yes_button = tk.Button(self.root, text="YES", command=self.on_yes_button)
        self.yes_button.grid(row=3, column=0)

        self.no_button = tk.Button(self.root, text="NO", command=self.on_no_button)
        self.no_button.grid(row=3, column=1)

        # Кнопка очистки холста
        self.clear_button = tk.Button(self.root, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=4, column=0, columnspan=2)

        # Кнопка для выгрузки весов
        self.export_button = tk.Button(self.root, text="Выгрузить веса", command=self.export_weights)
        self.export_button.grid(row=5, column=0)

        # Кнопка для загрузки весов
        self.import_button = tk.Button(self.root, text="Загрузить веса", command=self.import_weights)
        self.import_button.grid(row=5, column=1)

        # Кнопка для рандомизации весов
        self.randomize_button = tk.Button(self.root, text="Рандомизировать веса", command=self.randomize_weights)
        self.randomize_button.grid(row=6, column=0, columnspan=2)

        # Строка для отображения действий
        self.action_label = tk.Label(self.root, text="Действие: ")
        self.action_label.grid(row=7, column=0, columnspan=2)

    # Метод рисования на холсте
    def paint(self, event):
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE
        if 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE:
            self.draw.point((x, y), fill=0)
            self.update_pixel(x, y)
            self.recognize_symbol()

    # Обновление цвета пикселя
    def update_pixel(self, x, y):
        color = "black" if self.canvas_image.getpixel((x, y)) == 0 else "white"
        self.canvas.itemconfig(self.pixel_rects[x][y], fill=color)

    # Получение данных пикселей с холста
    def get_canvas_pixel_data(self):
        return [0 if pixel == 255 else 1 for pixel in self.canvas_image.getdata()]

    # Распознавание символа
    def recognize_symbol(self):
        pixels = self.get_canvas_pixel_data()
        prediction = self.neuron.predict(pixels)
        symbol = "λ" if prediction == 1 else "γ"
        self.recognized_symbol_text.delete(0, tk.END)
        self.recognized_symbol_text.insert(0, symbol)

    # Нажатие кнопки YES (символ угадан верно)
    def on_yes_button(self):
        self.action_label.config(text="Действие: отмечено как правильно распознанный")
        self.clear_canvas()

    # Нажатие кнопки NO (символ угадан неверно)
    def on_no_button(self):
        expected_output = 1 if self.recognized_symbol_text.get() == "λ" else 0
        pixels = self.get_canvas_pixel_data()
        self.neuron.train(pixels, expected_output)
        self.action_label.config(text="Действие: отмечено как неправильно распознанный")
        self.clear_canvas()

    # Очистка холста
    def clear_canvas(self):
        self.canvas_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.canvas_image)
        for x in range(CANVAS_SIZE):
            for y in range(CANVAS_SIZE):
                self.update_pixel(x, y)

    # Выгрузка весов в CSV
    def export_weights(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(["weights_data"])
                for weight in self.neuron.get_weights():
                    writer.writerow([weight])
            self.action_label.config(text=f"Действие: данные о весах выгружены в файл {file_path}")

    # Загрузка весов из CSV
    def import_weights(self):
        file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                next(reader)  # Пропустить заголовок
                weights = [float(row[0]) for row in reader]
                self.neuron.set_weights(weights)
            self.action_label.config(text=f"Действие: данные о весах загружены из файла {file_path}")

    # Рандомизация весов
    def randomize_weights(self):
        self.neuron.randomize_weights()
        self.action_label.config(text="Действие: веса рандомизированы")


# Создание и запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolRecognizerApp(root)
    root.mainloop()
