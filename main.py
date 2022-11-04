# import the opencv library
import cv2
import onnx
import onnxruntime as ort
import numpy as np


model_path = 'super_resolution.onnx'
model = onnx.load(model_path)
session = ort.InferenceSession(model.SerializeToString())
opencv_net = cv2.dnn.readNetFromONNX(model_path)
cap = cv2.VideoCapture(0)

def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    # img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    # img = np.transpose(img, axes=[2, 0, 1])
    # img = img.astype(np.float32)
    # img = np.expand_dims(img, axis=0)
    return img

def predict(path):
    # img = get_image(path, show=True)
    img = preprocess(frame)
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    # print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))

while 1:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    cv2.imshow('webcam', preprocess(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break