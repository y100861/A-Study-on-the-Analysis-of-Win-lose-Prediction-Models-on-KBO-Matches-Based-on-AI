# -*- coding: utf-8 -*-
# 나눔고딕 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mpl.rc("font", family="NanumGothic")
mpl.rcParams['axes.unicode_minus'] = False


# 데이터 불러오기
kbo_matches_pure_saber = pd.read_excel("/Users/yanghaejun/Desktop/논문/KBO 경기 데이터/kbo_matches.xlsx",
                                       sheet_name="all_kbo_matches_pure+saber")


# 종속변인 및 원-핫 인코딩이 필요한 변인 저장
y = kbo_matches_pure_saber["승1패0"]
team_names = kbo_matches_pure_saber["팀이름"]
home_away = kbo_matches_pure_saber["홈/어웨이"]


# 종속변인 및 원-핫 인코딩이 필요한 변인 원자료에서 제거
kbo_matches_pure_saber  = kbo_matches_pure_saber.drop(["팀이름", "홈/어웨이", "승1패0"], axis=1)
columns = kbo_matches_pure_saber.columns


# 기술통계
df = kbo_matches_pure_saber.describe()
df.transpose()


# 데이터 표준화
scaler = StandardScaler()
kbo_matches_pure_saber_scale = scaler.fit_transform(kbo_matches_pure_saber)
kbo_matches_pure_saber_scale = pd.DataFrame(kbo_matches_pure_saber_scale, columns=columns)


# 변인 간 상관관계
corr = kbo_matches_pure_saber_scale[pure_saber_columns].corr(method="pearson")


# 히트맵으로 시각화
plt.figure(figsize=(15, 15))
sns.heatmap(corr,
            fmt=".2f",
            cbar=True,
            annot_kws={"size": 12},
            annot=True,
            square=True)
plt.title("변인 간 상관관계",
          pad=30.0,
          size=25);


# 1차 다중공선성 분석 및 출력
vif_1 = pd.DataFrame()
vif_1["VIF Factor"] = [variance_inflation_factor(kbo_matches_pure_saber.values, i) for i in range(kbo_matches_pure_saber.shape[1])]
vif_1["변인"] = kbo_matches_pure_saber.columns


# 1차 다중공선성 분석에서 VIF 계수가 높은 변인 소거
kbo_matches_pure_saber_vif = kbo_matches_pure_saber[["홈런", "볼넷", "사구", "삼진", "잔루", "승계주자실점", "실책", "OPS", "승계주자실점률", "ERA", "WHIP", "K%", "BB%"]]


# 2차 다중공선성 분석
vif_2 = pd.DataFrame()
vif_2["VIF Factor"] = [variance_inflation_factor(kbo_matches_pure_saber_vif.values, i) for i in range(kbo_matches_pure_saber_vif.shape[1])]
vif_2["변인"] = kbo_matches_pure_saber_vif.columns


# 2차 다중공선성 분석에서 VIF 계수가 높은 변인 소거
kbo_matches_pure_saber_vif = kbo_matches_pure_saber[["볼넷", "사구", "삼진", "실책", "OPS", "승계주자실점률", "ERA", "K%", "BB%"]]


# 변인 선별 후 VIF 계수 출력
vif_3 = pd.DataFrame()
vif_3["VIF Factor"] = [variance_inflation_factor(kbo_matches_pure_saber_vif.values, i) for i in range(kbo_matches_pure_saber_vif.shape[1])]
vif_3["변인"] = kbo_matches_pure_saber_vif.columns


# 선별된 변인들 이름 저장
vif_columns = kbo_matches_pure_saber_vif.columns


# 데이터 표준화
scaler = StandardScaler()
kbo_matches_pure_saber_vif_scale = scaler.fit_transform(kbo_matches_pure_saber_vif)
kbo_matches_pure_saber_vif_scale = pd.DataFrame(kbo_matches_pure_saber_vif_scale, columns=vif_columns)


# 팀 이름 변인 원-핫 인코딩
team_names = pd.get_dummies(team_names, dtype=int)
team_names = team_names.rename(columns={"팀이름_KIA": "KIA",
                                        "팀이름_KT": "KT",
                                        "팀이름_LG": "LG",
                                        "팀이름_NC": "NC",
                                        "팀이름_SSG": "SSG",
                                        "팀이름_두산": "두산",
                                        "팀이름_롯데": "롯데",
                                        "팀이름_삼성": "삼성",
                                        "팀이름_키움": "키움",
                                        "팀이름_한화": "한화"})
kbo_matches_pure_saber_vif_scale = pd.concat([kbo_matches_pure_saber_vif_scale, team_names], axis=1)


# 홈/어웨이 변인 전처리
home_away = home_away.apply(lambda x: 1 if x == "홈" else 0)
kbo_matches_pure_saber_vif_scale = pd.concat([home_away, kbo_matches_pure_saber_vif_scale], axis=1)


# 승패변인 합치기
kbo_matches_pure_saber_vif_scale = pd.concat([y, kbo_matches_pure_saber_vif_scale], axis=1)


# 최종 데이터 세트 저장
kbo_matches_pure_saber_vif_scale.to_excel("/content/drive/MyDrive/논문(딥러닝)/최종데이터.xlsx")

===================================== 최종 데이터 세트 완성 =====================================
