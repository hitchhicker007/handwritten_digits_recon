import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

model = load_model('mnist_trained.h5')

img = cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(28,28))
img = img.reshape(-1,28,28,1)
img = img / 255

print(model.predict_classes(img))