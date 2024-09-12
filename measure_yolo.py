from ultralytics import YOLO
import glob
import numpy as np
import os

# Оценка модели
model = YOLO('runs/detect/train3/weights/best.pt')

test_images_path = r'materials\calibrate\Cam2_6.jpg'  # Путь к изображению
test_images = glob.glob(test_images_path)

# Путь к файлу с коэффициентом
scale_factor_file = 'scale_factor.txt'

# Функция для чтения коэффициента из файла
def read_scale_factor(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            try:
                # Извлекаем число из строки, если оно есть
                scale_factor = float(line.split(':')[1].strip())
                return scale_factor
            except (IndexError, ValueError):
                print("Ошибка при чтении коэффициента. Используется коэффициент по умолчанию.")
                return 1
    else:
        print(f"Файл {file_path} не найден. Используется коэффициент по умолчанию.")
        return 1

# Чтение коэффициента
scale_factor = read_scale_factor(scale_factor_file)

# Функция для расчета евклидова расстояния
def calculate_pixel_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Визуализация на тестовых данных
for img_path in test_images:
    # Запуск модели на тестовых данных
    results = model(img_path, conf=0.25)
    
    # Проход по bounding boxes
    for result in results:
        boxes = result.boxes.xyxy  # Получаем координаты боксов
        
        if len(boxes) >= 2:
            # Выбираем первые два бокса для расчета расстояния
            x1_center = (boxes[0][0] + boxes[0][2]) / 2  # Центр первого бокса (по x)
            y1_center = (boxes[0][1] + boxes[0][3]) / 2  # Центр первого бокса (по y)

            x2_center = (boxes[1][0] + boxes[1][2]) / 2  # Центр второго бокса (по x)
            y2_center = (boxes[1][1] + boxes[1][3]) / 2  # Центр второго бокса (по y)

            # Рассчитываем пиксельное расстояние
            pixel_distance = calculate_pixel_distance((x1_center, y1_center), (x2_center, y2_center))
            
            # Переводим в реальное расстояние
            real_distance = pixel_distance * scale_factor

            print(f"Расстояние между двумя маркерами: {real_distance:.2f} мм")
        else:
            print("Недостаточно маркеров для расчета расстояния.")
