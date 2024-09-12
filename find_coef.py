from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Загрузка модели YOLO
model = YOLO('runs/detect/train3/weights/best.pt')

# Путь к изображению
img_path = r'materials\mark_seti\markers_set_5mm_diam'

# Выполнение детекции
results = model(img_path, conf=0.6)

# Константы
KNOWN_DISTANCE = 10  # Расстояние между точками в мм

# Функция для вычисления расстояния между двумя точками
def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

# Функция для нахождения двух ближайших точек
def find_closest_points(points):
    min_distance = float('inf')
    closest_pair = (None, None)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = calculate_distance(points[i], points[j])
            if dist < min_distance:
                min_distance = dist
                closest_pair = (points[i], points[j])
    return closest_pair, min_distance

# Обработка результатов
for result in results:
    # Извлечение bounding boxes
    boxes = result.boxes
    centers = []
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            centers.append(center)

        if len(centers) >= 2:
            # Найти две ближайшие точки
            (pt1, pt2), distance_pixels = find_closest_points(centers)
            
            # Вычисление масштаба
            scale_factor = KNOWN_DISTANCE / distance_pixels
            print(f"Distance (pixels): {distance_pixels:.2f}")
            print(f"Scale factor: {scale_factor:.4f}")
            
            # # Применение масштаба
            # for i, (x, y) in enumerate(centers):
            #     print(f"Point {i + 1}: ({x:.2f}, {y:.2f}) in pixels")
        else:
            print("Not enough circles detected.")
    else:
        print("No detections found.")

    # Рисование боксов и центров
    img = result.plot()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
