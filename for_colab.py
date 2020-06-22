from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

pretrained_model = "https://github.com/tae898/age-gender-estimation/releases/download/v0.1/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                        file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

depth = 16
k = 8
margin = 0.4

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)
print(f"model loaded!")

# for img in image_generator:
img_BGR = cv2.imread('photo.jpg')
print(f"imaged 'photo.jpg' loaded!")
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = np.shape(img_RGB)

# detect faces using dlib detector
detected = detector(img_RGB, 1)
faces = np.empty((len(detected), img_size, img_size, 3))

if len(detected) > 0:
    for i, d in enumerate(detected):
        x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), img_w - 1)
        yw2 = min(int(y2 + margin * h), img_h - 1)
        cv2.rectangle(img_BGR, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
        faces[i, :, :, :] = cv2.resize(img_BGR[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

# predict ages and genders of the detected faces
results = model.predict(faces)
predicted_genders = results[0]
ages = np.arange(0, 101).reshape(101, 1)
predicted_ages = results[1].dot(ages).flatten()

# draw results
for i, d in enumerate(detected):
    label = "{}, {}".format(int(predicted_ages[i]),
                            "M" if predicted_genders[i][0] < 0.5 else "F")
    draw_label(img_BGR, (d.left(), d.top()), label)

cv2.imwrite('photo_annotated.jpg', img_BGR)
print(f"An annotated image saved as 'photo_annotated.jpg'")
