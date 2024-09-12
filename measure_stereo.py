import cv2
import numpy as np
import os

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
    
def undistort_image(image, camera_matrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs)

def calculate_distance_3d(point1, point2, P1, P2, scale_factor):
    # Преобразование точек в формат, подходящий для триангуляции
    points_2d_1 = np.array([[point1[0], point1[1]], [point2[0], point2[1]]], dtype=np.float32).T
    points_2d_2 = np.array([[point1[0], point1[1]], [point2[0], point2[1]]], dtype=np.float32).T

    # Триангуляция точек
    points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_1, points_2d_2)
    
    # Преобразование гомогенных координат в 3D
    points_4d = points_4d_hom[:3] / points_4d_hom[3]

    # Расчет расстояния между точками
    distance = np.linalg.norm(points_4d[:, 0] - points_4d[:, 1])
    return distance * scale_factor

# Чтение коэффициента
scale_factor = read_scale_factor(scale_factor_file)

# Параметры камеры
fx1, fy1 = 80, 80  # Фокусное расстояние 80 мм
cx1, cy1 = 36.17 / 2, 24.11 / 2  # Оптический центр
dist_coeffs1 = np.array([0.04, 0, 0, 0, 0], dtype=np.float64)  # Искажения

camera_matrix1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]], dtype=np.float64)
camera_matrix2 = camera_matrix1  # Одинаковые параметры для обеих камер

# Параметры съемки
T = np.array([412, 0, 0], dtype=np.float64)  # Расстояние между камерами 412 мм

# Углы камер
angle_left = np.deg2rad(17.87)
angle_right = np.deg2rad(17.33)

# Матрицы поворота для обеих камер
R_left = np.array([[np.cos(angle_left), 0, np.sin(angle_left)],
                   [0, 1, 0],
                   [-np.sin(angle_left), 0, np.cos(angle_left)]], dtype=np.float64)

R_right = np.array([[np.cos(angle_right), 0, -np.sin(angle_right)],
                    [0, 1, 0],
                    [np.sin(angle_right), 0, np.cos(angle_right)]], dtype=np.float64)

# Общая матрица поворота между камерами
R = R_right @ R_left.T

# Коррекция искажений
img1 = cv2.imread(r'materials\calibrate\Cam1_4.jpg')
img2 = cv2.imread(r'materials\calibrate\Cam2_4.jpg')

img1_undistorted = undistort_image(img1, camera_matrix1, dist_coeffs1)
img2_undistorted = undistort_image(img2, camera_matrix2, dist_coeffs1)

# Выбор точек на изображении
def select_points(event, x, y, flags, param):
    global points, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)

points = []
img_copy = img1_undistorted.copy()

cv2.imshow("Image", img_copy)
cv2.setMouseCallback("Image", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Проверка, что выбраны две точки
if len(points) == 2:
    point1, point2 = points
    print(f"Точка 1: {point1}, Точка 2: {point2}")

    # Подготовка матриц проекции
    P1 = np.hstack((camera_matrix1, np.zeros((3, 1))))
    P2 = np.hstack((camera_matrix2 @ R, -camera_matrix2 @ R @ T.reshape(-1, 1)))

    # Расчет расстояния в 3D
    distance = calculate_distance_3d(point1, point2, P1, P2, scale_factor)
    print(f"Расстояние между точками: {distance:.2f} мм")
else:
    print("Выберите две точки на изображении.")
