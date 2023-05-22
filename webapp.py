import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import MTCNN

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import threading
import av


def cvt_age(age):
    return str(age)+"0+"


def cvt_ethnicity(eth):
    if eth == 0:
        return "white"
    if eth == 1:
        return "black"
    if eth == 2:
        return "asian"
    if eth == 3:
        return "indian"
    return "unknown"


def cvt_gender(gen):
    if gen == 0:
        return "male"
    return "female"

# %%


class MultiLabelCNN(nn.Module):
    def __init__(self, age_features, ethnicity_features, gender_features):
        super().__init__()
        '''
        self.model = timm.create_model('dm_nfnet_f0', pretrained = False)
        n_features = self.model.num_features
        self.age_classifier = ClassifierHead(n_features, age_features)
        self.eth_classifier = ClassifierHead(n_features, ethnicity_features)
        self.gen_classifier = ClassifierHead(n_features, gender_features)
        '''
        self.cnnModel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 24
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 12
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 6
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 3
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # flatten
        )

        self.dnnModel = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )

        self.age_classifier = nn.Linear(32, age_features)
        self.eth_classifier = nn.Linear(32, ethnicity_features)
        self.gen_classifier = nn.Linear(32, gender_features)

    def forward(self, x):
        '''
        output = self.model.forward_features(x)
        age = self.age_classifier(output)
        eth = self.eth_classifier(output)
        gen = self.gen_classifier(output)
        '''
        output = self.cnnModel(x)
        output = output.squeeze()
        output = self.dnnModel(output)

        age = self.age_classifier(output)
        eth = self.eth_classifier(output)
        gen = self.gen_classifier(output)
        return age, eth, gen


model = MultiLabelCNN(12, 5, 2)
model.load_state_dict(torch.load(
    "./predmodel.pth", map_location=torch.device('cpu')))
model.eval()

# %%


def predictClass(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    transform = transforms.Compose({
        transforms.ToTensor()
    })
    img = img.reshape(48, 48, 1)
    img = transform(img)

    predict = model(img)

    age = torch.argmax(predict[0]).item()
    eth = torch.argmax(predict[1]).item()
    gen = torch.argmax(predict[2]).item()

    return age, eth, gen

# %%


class FaceDetector(object):

    def __init__(self, mtcnn) -> None:
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, pclass):

        for box, prob in zip(boxes, probs):
            # cropped_frame = frame[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
                box[2]), int(box[3])), (0, 0, 255), thickness=2)
            cv2.putText(frame, pclass, (int(box[2]), int(
                box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=2)
        #         cv2.putText(frame, str(predictClass(cropped_frame, model)), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # try:
        #     for box, prob in zip(boxes, probs):
        #         cropped_frame = frame[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]

        #         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness=2)
        #         cv2.putText(frame, str(predictClass(cropped_frame, model)), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #         # cv2.putText(frame, "x", (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        #         # frame = frame[int(box[0]):int(box[2]), int(box[1]), int(box[3])]
        #         # cv2.putText(frame, str(prob), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        #         # cv2.circle(frame, tuple(Id[0]), 5, (0, 0, 255), -1)
        #         # cv2.circle(frame, tuple(Id[1]), 5, (0, 0, 255), -1)
        #         # cv2.circle(frame, tuple(Id[2]), 5, (0, 0, 255), -1)
        #         # cv2.circle(frame, tuple(Id[3]), 5, (0, 0, 255), -1)
        #         # cv2.circle(frame, tuple(Id[4]), 5, (0, 0, 255), -1)
        # except:
        #     pass

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        pclass = "predicting..."
        frame_count = 0
        predict_threshold = 120
        can_age = []
        can_eth = []
        can_gen = []

        while True:
            ret, frame = cap.read()
            try:
                boxes, probs = self.mtcnn.detect(frame)

                if frame_count < predict_threshold:
                    pclass = "predicting..." + \
                        str(round(frame_count/predict_threshold * 100, 2)) + "%"
                    cropped_framed = frame[int(boxes[0][1]):int(
                        boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
                    age, eth, gen = predictClass(cropped_framed, model)
                    can_age.append(age)
                    can_eth.append(eth)
                    can_gen.append(gen)

                elif frame_count == predict_threshold:
                    predicted_age = cvt_age(
                        max(set(can_age), key=can_age.count))
                    predicted_eth = cvt_ethnicity(
                        max(set(can_eth), key=can_eth.count))
                    predicted_gen = cvt_gender(
                        max(set(can_gen), key=can_gen.count))
                    pclass = predicted_age + " " + predicted_eth + " " + predicted_gen

                frame = self._draw(frame, boxes, probs, pclass)

            except:
                pass

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()


def draw(frame, boxes, probs, pclass):

    for box, prob in zip(boxes, probs):
        # cropped_frame = frame[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
            box[2]), int(box[3])), (255, 255, 0), thickness=2)
        cv2.putText(frame, pclass, (int(box[2]), int(
            box[3])), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)

    return frame


mtcnn = MTCNN()

lock = threading.Lock()
state = {
    "frame_count": 0,
    "pclass": "predicting...",
    "predict_threshold": 120,
    "can_age": [],
    "can_eth": [],
    "can_gen": []

}


def callback(vframe):

    frame = vframe.to_ndarray(format="bgr24")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with lock:
        frame_count = state["frame_count"]
        pclass = state["pclass"]
        predict_threshold = state["predict_threshold"]

    try:
        boxes, probs = mtcnn.detect(frame)

        if frame_count < predict_threshold:
            pclass = "predicting..." + \
                str(round(frame_count/predict_threshold * 100, 2)) + "%"
            cropped_framed = frame[int(boxes[0][1]):int(
                boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
            age, eth, gen = predictClass(cropped_framed, model)
            with lock:
                state["can_age"].append(age)
                state["can_eth"].append(eth)
                state["can_gen"].append(gen)

        elif frame_count == predict_threshold:
            with lock:
                can_age = state["can_age"]
                can_eth = state["can_eth"]
                can_gen = state["can_gen"]
            predicted_age = cvt_age(
                max(set(can_age), key=can_age.count))
            predicted_eth = cvt_ethnicity(
                max(set(can_eth), key=can_eth.count))
            predicted_gen = cvt_gender(
                max(set(can_gen), key=can_gen.count))

            with lock:
                state["pclass"] = predicted_age + " " + \
                    predicted_eth + " " + predicted_gen

        # cropped_framed = frame[int(boxes[0][1]):int(
        #     boxes[0][3]), int(boxes[0][0]):int(boxes[0][2])]
        # age, eth, gen = predictClass(cropped_framed, model)
        # pclass = f'{str(cvt_age(age))} {str(cvt_ethnicity(eth))} {str(cvt_gender(gen))}'
        frame = draw(frame, boxes, probs, pclass)

    except:
        pass

    with lock:
        state["frame_count"] = state["frame_count"] + 1

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return av.VideoFrame.from_ndarray(frame, format="bgr24")


st.title("Face Appearance Prediction")
state["predict_threshold"] = st.slider(
    "Prediction Frames", min_value=30, max_value=240, step=1, value=30)
FRAME_WINDOW = st.image([])


webrtc_streamer(key="example", video_frame_callback=callback, video_html_attrs=VideoHTMLAttributes(
    autoPlay=True, controls=False, style={"width": "100%"}, muted=True))

# %%
