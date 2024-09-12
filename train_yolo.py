import cv2
import numpy as np
import os

# Путь к изображениям и аннотациям
folder_path = r'C:\Users\vladt\Downloads\Hacaton.v1i.yolov8\calibrate_stereo'

# Задание размеров маркерной сети
marker_diameter = 5.0  # мм

# Параметры для стереокалибровки
object_points = []  # Мировые координаты маркеров
image_points_left = []  # Точки на изображениях с камеры 1
image_points_right = []  # Точки на изображениях с камеры 2

# Функция для отображения изображения с помеченными точками
def show_markers(image_path, markers):
    img = cv2.imread(image_path)
    for (x, y) in markers:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # Рисуем окружности
    cv2.imshow("Markers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Функция для чтения аннотаций YOLO
def read_yolo_annotations(image_path, annotations_path):
    with open(annotations_path, 'r') as file:
        lines = file.readlines()
    img = cv2.imread(image_path)
    img_height, img_width, _ = img.shape
    points = []
    for line in lines:
        parts = line.strip().split()
        x_center, y_center = float(parts[1]), float(parts[2])
        # Преобразование координат из относительных в абсолютные
        x = x_center * img_width
        y = y_center * img_height
        points.append([x, y])
    return np.array(points, dtype=np.float32)

# Загружаем изображения и аннотации
for i in range(2, 3):
    cam1_image_path = os.path.join(folder_path, f"Cam1_{i}_jpg.rf.3f5c906917a1c0a3753c62e34dfe9af9.jpg")
    cam1_annotations_path = os.path.join(folder_path, f"Cam1_{i}_jpg.rf.3f5c906917a1c0a3753c62e34dfe9af9.txt")
    cam2_image_path = os.path.join(folder_path, f"Cam2_{i}_jpg.rf.cd35f0aced9c1c24b1244fb495f39d71.jpg")
    cam2_annotations_path = os.path.join(folder_path, f"Cam2_{i}_jpg.rf.cd35f0aced9c1c24b1244fb495f39d71.txt")

    markers_left = read_yolo_annotations(cam1_image_path, cam1_annotations_path)
    markers_right = read_yolo_annotations(cam2_image_path, cam2_annotations_path)

    if markers_left is not None and markers_right is not None:
        # Отображаем изображения с найденными маркерами
        show_markers(cam1_image_path, markers_left)
        show_markers(cam2_image_path, markers_right)

        # Предположим, что маркеры расположены в известной сетке, например 5 мм между центрами кругов
        objp = np.zeros((len(markers_left), 3), np.float32)
        objp[:, :2] = np.mgrid[0:len(markers_left), 0:1].T.reshape(-1, 2) * marker_diameter

        object_points.append(objp)
        image_points_left.append(markers_left.reshape(-1, 1, 2))  # Преобразуем в формат (N, 1, 2)
        image_points_right.append(markers_right.reshape(-1, 1, 2))  # Преобразуем в формат (N, 1, 2)

# Определение размера изображения из первого изображения
img = cv2.imread(cam1_image_path)
if img is not None:
    image_size = (img.shape[1], img.shape[0])  # Ширина и высота изображения
else:
    raise Exception("Не удалось загрузить изображение для определения размера.")

# Проверьте размер данных
print("Формат object_points:", [op.shape for op in object_points])
print("Формат image_points_left:", [ip.shape for ip in image_points_left])
print("Формат image_points_right:", [ip.shape for ip in image_points_right])
print("Размер изображения:", image_size)

# Убедитесь, что все списки имеют одинаковую длину
if len(object_points) == len(image_points_left) == len(image_points_right):
    print("Количество точек совпадает.")
else:
    raise ValueError("Количество точек не совпадает между object_points и image_points")

# После сбора данных производим стереокалибровку
try:
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        object_points, image_points_left, image_points_right, None, None, None, None, image_size)

    # Результат калибровки
    print("Матрица камеры 1:\n", mtx_left)
    print("Матрица камеры 2:\n", mtx_right)
    print("Вращение R:\n", R)
    print("Смещение T:\n", T)

except cv2.error as e:
    print("Ошибка при выполнении стереокалибровки:", e)

# Загрузка изображений
img1 = cv2.imread(r'C:\Users\vladt\Downloads\Hacaton.v1i.yolov8\calibrate_stereo\Cam1_2_jpg.rf.3f5c906917a1c0a3753c62e34dfe9af9.jpg')
img2 = cv2.imread(r'C:\Users\vladt\Downloads\Hacaton.v1i.yolov8\calibrate_stereo\Cam2_2_jpg.rf.cd35f0aced9c1c24b1244fb495f39d71.jpg')

# Преобразование изображения в черно-белое
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Выполнение стереоректикализации
R1, R2, P1, P2, Q, valid_pix_ROI1, valid_pix_ROI2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right, img1.shape[:2], R, T, flags=cv2.CALIB_ZERO_DISPARITY)

# Преобразование изображений в выровненное представление
map1x, map1y = cv2.initUndistortRectifyMap(mtx_left, dist_left, R1, P1, img1.shape[:2], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_right, dist_right, R2, P2, img2.shape[:2], cv2.CV_32FC1)
rectified_img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rectified_img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# Примеры координат точек на изображениях
points1 = np.array([[100, 150], [200, 250]], dtype=np.float32)  # Пример координат на img1
points2 = np.array([[110, 160], [210, 260]], dtype=np.float32)  # Пример координат на img2

# Проекционные матрицы
projMat1 = np.hstack((P1[:, :3], P1[:, 3:]))
projMat2 = np.hstack((P2[:, :3], P2[:, 3:]))

# Триангуляция точек
points4D = cv2.triangulatePoints(projMat1, projMat2, points1.T, points2.T)

# Преобразование гомогенных координат в 3D
points3D = points4D[:3] / points4D[3]
print("3D координаты точек:\n", points3D.T)

# Объединение изображений
concatenated_img = np.hstack((rectified_img1, rectified_img2))

# Визуализация изображения и выбор точек
def show_points_and_distance(img1, img2, points1, points2):
    concatenated_img = np.hstack((img1, img2))
    for (x, y) in points1:
        cv2.circle(concatenated_img, (int(x), int(y)), 5, (0, 255, 0), -1)
    for (x, y) in points2:
        cv2.circle(concatenated_img, (int(x + img1.shape[1]), int(y)), 5, (0, 0, 255), -1)
    
    # Выбор двух точек
    pt1 = (int(points1[0][0]), int(points1[0][1]))
    pt2 = (int(points2[0][0] + img1.shape[1]), int(points2[0][1]))

    # Рисование линии между точками и вычисление расстояния
    cv2.line(concatenated_img, pt1, pt2, (255, 0, 0), 2)
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    cv2.putText(concatenated_img, f"Distance: {distance:.2f} px", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Points and Distance", concatenated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_points_and_distance(rectified_img1, rectified_img2, points1, points2)
