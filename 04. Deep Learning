# -*- coding: utf-8 -*-
# 나눔고딕 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
import shap
mpl.rc("font", family="NanumGothic")
mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='NanumBarunGothic')

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model


# SEED 고정
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# 데이터 불러오기(변인들이 선별된 데이터)
kbo_matches_pure_saber_vif_scale = pd.read_excel("/content/drive/MyDrive/논문(딥러닝)/최종데이터.xlsx")


# Pilot 모델 생성
## 데이터 랜덤 추출(20%)
pilot_data = kbo_matches_pure_saber_vif_scale.sample(frac=0.2, replace=True, random_state=1)
pilot_data = pilot_data.reset_index().drop(["index"], axis=1)

## pilot 종속변인 및 팀이름, 홈/어웨이
y_pilot = pilot_data["승1패0"]
pilot_data = pilot_data.drop("승1패0", axis=1)

## Pilot 모델을 위한 학습, 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(pilot_data, y_pilot, test_size=0.2, random_state=1)

------------------------------ Pilot 1 ------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
def pilot1_deep_model():
    input = Input(shape=(20,), name="input")
    hidden1 = Dense(15, activation="relu", name="hidden1")(input)
    hidden2 = Dense(9, activation="relu", name="hidden2")(hidden1)
    hidden3 = Dense(4, activation="relu", name="hidden3")(hidden2)
    output = Dense(1, activation="sigmoid", name="output")(hidden3)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

pilot1_deep_model = pilot1_deep_model()
pilot1_deep_model.fit(X_train, y_train, epochs=200)
pilot1_deep_model.evaluate(X_test, y_test)
------------------------------ Pilot 1 ------------------------------

------------------------------ Pilot 2 ------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
def pilot2_deep_model():
    input = Input(shape=(20,), name="input")
    hidden1 = Dense(15, activation="relu", name="hidden1")(input)
    hidden2 = Dense(9, activation="relu", name="hidden2")(hidden1)
    output = Dense(1, activation="sigmoid", name="output")(hidden2)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

pilot2_deep_model = pilot2_deep_model()
pilot2_deep_model.fit(X_train, y_train, epochs=200)
pilot2_deep_model.evaluate(X_test, y_test)
------------------------------ Pilot 2 ------------------------------

------------------------------ Pilot 3 ------------------------------
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
def pilot3_deep_model():
    input = Input(shape=(20,), name="input")
    hidden1 = Dense(12, activation="relu", name="hidden1")(input)
    hidden2 = Dense(5, activation="relu", name="hidden2")(hidden1)
    output = Dense(1, activation="sigmoid", name="output")(hidden2)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

pilot3_deep_model = pilot3_deep_model()
pilot3_deep_model.fit(X_train, y_train, epochs=200)
pilot3_deep_model.evaluate(X_test, y_test)
------------------------------ Pilot 3 ------------------------------
===================== Deep Learning 모델 구성 완료 =====================


# 데이터 다시 불러오기
kbo_matches_pure_saber_vif_scale = pd.read_excel("/content/drive/MyDrive/논문(딥러닝)/최종데이터.xlsx")


# 종속변인(승패)
y = kbo_matches_pure_saber_vif_scale["승1패0"]
kbo_matches_pure_saber_vif_scale = kbo_matches_pure_saber_vif_scale.drop("승1패0", axis=1)


# Deep Learning에 용이하도록 numpy 배열로 변환
kbo_matches_pure_saber_vif_scale = kbo_matches_pure_saber_vif_scale.to_numpy()
kbo_matches_pure_saber_vif_scale[0]


# KFold
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# SHAP를 사용하기 위해 KFold 교차 검증 과정에서 사용된 데이터들의 인덱스 저장
ix_training, ix_test = [], []

# 각 회차에서 빈 리스트에 인덱스 저장
for fold in kfold.split(kbo_matches_pure_saber_vif_scale):
    ix_training.append(fold[0]), ix_test.append(fold[1])


============================== Deep Learning ==============================
# 채택된 Pilot3 모델의 구조
def deep_model():
    input = Input(shape=(20,), name="input")
    hidden1 = Dense(12, activation="relu", name="hidden1")(input)
    hidden2 = Dense(5, activation="relu", name="hidden2")(hidden1)
    output = Dense(1, activation="sigmoid", name="output")(hidden2)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

# 독립변인
feature_names = ["홈/어웨이",
                 "볼넷", "사구", "삼진", "실책", "OPS", "승계주자실점률", "ERA", "K%", "BB%",
                 "KIA", "KT", "LG", "NC", "SSG", "두산", "롯데", "삼성", "키움", "한화"]

# 변인 중요도 리스트
SHAP_values_per_fold = []
feature_imp_list = []

# 평가지표 리스트
train_accuracy_list = []
train_f1_list = []
train_precision_list = []
train_recall_list = []
test_accuracy_list = []
test_f1_list = []
test_precision_list = []
test_recall_list = []

# KFold 교차검증
for train_idx, test_idx in kfold.split(kbo_matches_pure_saber_vif_scale):

    # 학습용/평가용 데이터 분할
    X_train_fold = kbo_matches_pure_saber_vif_scale[train_idx]
    X_test_fold = kbo_matches_pure_saber_vif_scale[test_idx]
    y_train_fold = y[train_idx]
    y_test_fold = y[test_idx]

    # 딥러닝 학습 및 예측 수행
    model_deep = deep_model()
    model_deep.fit(X_train_fold, y_train_fold, epochs=200, validation_data=(X_test_fold, y_test_fold))
    y_train_predict = model_deep.predict(X_train_fold).round()
    y_test_predict = model_deep.predict(X_test_fold).round()

    # 학습 데이터 예측 평가지표
    train_accuracy = round(accuracy_score(y_train_fold, y_train_predict), 3)
    train_accuracy_list.append(train_accuracy)

    train_f1 = round(f1_score(y_train_fold, y_train_predict), 3)
    train_f1_list.append(train_f1)

    train_precision = round(precision_score(y_train_fold, y_train_predict), 3)
    train_precision_list.append(train_precision)

    train_recall = round(recall_score(y_train_fold, y_train_predict), 3)
    train_recall_list.append(train_recall)

    # 테스트 데이터 예측 평가지표
    test_accuracy = round(accuracy_score(y_test_fold, y_test_predict), 3)
    test_accuracy_list.append(test_accuracy)

    test_f1 = round(f1_score(y_test_fold, y_test_predict), 3)
    test_f1_list.append(test_f1)

    test_precision = round(precision_score(y_test_fold, y_test_predict), 3)
    test_precision_list.append(test_precision)

    test_recall = round(recall_score(y_test_fold, y_test_predict), 3)
    test_recall_list.append(test_recall)

    # SHAP
    explainer = shap.DeepExplainer(model=model_deep, data=X_test_fold)
    shap_values = explainer.shap_values(X_test_fold)
    for shap_value in shap_values[0]:
        SHAP_values_per_fold.append(shap_value)

    # 변인 중요도
    for i in range(X_test_fold.shape[1]):
        feature_imp = np.mean(np.abs(shap_values[0][:, i]))
        feature_imp_list.append(feature_imp)

============================== Deep Learning ==============================


# Accuracy, Precision, Recall, F1 저장(Train data)
accuracy_score_mean = round(np.mean(train_accuracy_list, axis=0), 3)
precision_score_mean = round(np.mean(train_precision_list, axis=0), 3)
recall_score_mean = round(np.mean(train_recall_list, axis=0), 3)
f1_score_mean = round(np.mean(train_f1_list, axis=0), 3)

# 각 점수의 평균값 리스트에 저장(Train data)
train_accuracy_list.append(accuracy_score_mean)
train_precision_list.append(precision_score_mean)
train_recall_list.append(recall_score_mean)
train_f1_list.append(f1_score_mean)

# 출력(Train data)
order = ["1차", "2차", "3차", "4차", "5차", "평균"]
index = ["Accuracy", "Precision", "Recall", "F1"]
validation_df = pd.DataFrame(data=[train_accuracy_list, train_precision_list, train_recall_list, train_f1_list], columns=order, index=index)


# Accuracy, Precision, Recall, F1 저장(Test data)
accuracy_score_mean = round(np.mean(test_accuracy_list, axis=0), 3)
precision_score_mean = round(np.mean(test_precision_list, axis=0), 3)
recall_score_mean = round(np.mean(test_recall_list, axis=0), 3)
f1_score_mean = round(np.mean(test_f1_list, axis=0), 3)

# 각 점수의 평균값 리스트에 저장(Test data)
test_accuracy_list.append(accuracy_score_mean)
test_precision_list.append(precision_score_mean)
test_recall_list.append(recall_score_mean)
test_f1_list.append(f1_score_mean)

# 출력(Test data)
order = ["1차", "2차", "3차", "4차", "5차", "평균"]
index = ["Accuracy", "Precision", "Recall", "F1"]
validation_df = pd.DataFrame(data=[test_accuracy_list, test_precision_list, test_recall_list, test_f1_list], columns=order, index=index)


# 데이터가 numpy 배열로 변환되어 있으므로 다시 데이터 불러옴
kbo_matches_pure_saber_vif_scale = pd.read_excel("/content/drive/MyDrive/논문(딥러닝)/최종데이터.xlsx")
y = kbo_matches_pure_saber_vif_scale["승1패0"]
kbo_matches_pure_saber_vif_scale = kbo_matches_pure_saber_vif_scale.drop("승1패0", axis=1)


# SHAP 절대적 영향도 수치 출력
feature_imp_list = np.array(feature_imp_list).reshape(5, 20)
feature_imp_df = pd.DataFrame(feature_imp_list)
feature_imp_list_final = []
for i in range(feature_imp_list.shape[1]):
    feature_imp_list_final.append(np.mean(feature_imp_df.iloc[:, i]))
for i in range(len(feature_imp_list_final)):
    print("{}의 중요도: {:.4f}".format(feature_names[i], feature_imp_list_final[i]))


# SHAP 절대적 영향도 그래프 출력
shap.summary_plot(np.array(SHAP_values_per_fold), features=kbo_matches_pure_saber_vif_scale.reindex(new_index), plot_type="bar")


# SHAP 긍/부정 영향도 그래프 출력
new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
shap.summary_plot(np.array(SHAP_values_per_fold), features=kbo_matches_pure_saber_vif_scale.reindex(new_index));
