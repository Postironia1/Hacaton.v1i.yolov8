import cv2
import numpy as np

# Шаг 1: Коррекция искажений (если необходимо)
def undistort_image(image, camera_matrix, dist_coeffs):
    return cv2.undistort(image, camera_matrix, dist_coeffs)

# Шаг 2: Нахождение совпадающих точек и вычисление гомографии
def find_keypoints_and_matches(img1, img2):
    # Инициализация детектора ORB
    orb = cv2.ORB_create()

    # Нахождение ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Нахождение совпадений дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Сортировка совпадений по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    # Получение точек для вычисления гомографии
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts

def rectify_images(img1, img2, camera_matrix1, camera_matrix2, dist_coeffs1, dist_coeffs2):
    src_pts, dst_pts = find_keypoints_and_matches(img1, img2)

    # Вычисление гомографии
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

    # Применение гомографии для коррекции изображения
    img1_rectified = cv2.warpPerspective(img1, np.eye(3), (img1.shape[1], img1.shape[0]))
    img2_rectified = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

    return img1_rectified, img2_rectified

# Шаг 3: Расчет матрицы Q
def compute_Q(camera_matrix1, camera_matrix2, R, T):
    Q = np.array([[1, 0, 0, -camera_matrix1[0, 2]],
                  [0, 1, 0, -camera_matrix1[1, 2]],
                  [0, 0, 0, camera_matrix1[0, 0]],
                  [0, 0, -1 / T[0], (camera_matrix1[0, 2] - camera_matrix2[0, 2]) / T[0]]], dtype=np.float64)
    return Q

# Шаг 4: Триангуляция и расчет 3D расстояния
def calculate_distance_3d(point1, point2, P1, P2):
    # Преобразование точек в формат, подходящий для триангуляции
    points_2d_1 = np.array([[point1[0], point1[1]], [point2[0], point2[1]]], dtype=np.float32).T
    points_2d_2 = np.array([[point1[0], point1[1]], [point2[0], point2[1]]], dtype=np.float32).T

    # Триангуляция точек
    points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_1, points_2d_2)
    
    # Преобразование гомогенных координат в 3D
    points_4d = points_4d_hom[:3] / points_4d_hom[3]

    # Расчет расстояния между точками
    distance = np.linalg.norm(points_4d[:, 0] - points_4d[:, 1])
    return distance * 0.0881

# Параметры камеры
fx1, fy1 = 80, 80  # Фокусное расстояние 80 мм
cx1, cy1 = 36.17 / 2, 24.11 / 2  # Оптический центр
dist_coeffs1 = np.array([0.04, 0, 0, 0, 0], dtype=np.float64)  # Искажения

camera_matrix1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]], dtype=np.float64)
camera_matrix2 = camera_matrix1  # Одинаковые параметры для обеих камер

# Параметры съёмки
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

# Ректификация изображений
img1_rectified, img2_rectified = rectify_images(img1_undistorted, img2_undistorted, camera_matrix1, camera_matrix2, dist_coeffs1, dist_coeffs1)

# Выбор точек на изображении
def select_points(event, x, y, flags, param):
    global points, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)

points = []
img_copy = img1_rectified.copy()

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
    distance = calculate_distance_3d(point1, point2, P1, P2)
    print(f"Расстояние между точками: {distance} мм")
else:
    print("Выберите две точки на изображении.")
