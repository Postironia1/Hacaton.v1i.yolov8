from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import glob

# Оценка модели
model = YOLO('runs/detect/train3/weights/best.pt')

test_images_path = r'test\images\*.jpg'  # Adjust the file extension if needed
test_images = glob.glob(test_images_path)

# Визуализация предиктов на тестовом датасете
for img_path in test_images:
    # Run the model on the test image
    results = model(img_path, conf=0.25)
    print(results)

    # Retrieve and plot the image with predictions
    for result in results:
        print(result[0].boxes)
        img = result.plot()  # This plots the bounding boxes on the image

        # Display the image with bounding boxes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()