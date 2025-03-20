# 스마트팩토리 제품 품질 분류
--------------------------------------
스마트 팩토리의 공정 과정의 제품 데이터를 활용하여 제품 품질을 예측하는 경진대회입니다.

## 🛠️ 기술 스택
- **언어**: Python  
- **라이브러리**: Scikit-learn, XGBoost, CatBoost, SVM
- **데이터 분석 & 처리**: Pandas, NumPy
----------------------------------
## 블로그 정리
XGBoost를 선택한 이유
(https://velog.io/@kimminyoung0/ML-XGBoost-LightGBM-CatBoost-비교)(https://velog.io/@kimminyoung0/ML-XGBoost-LightGBM-CatBoost-비교)

## Dataset Info.
### train.csv
- PRODUCT_ID : 제품의 고유 ID
- Y_Class : 제품 품질 상태(Target) - 0 : 적정 기준 미달 (부적합) / 1 : 적합 / 2 : 적정 기준 초과 (부적합)
- Y_Quality : 제품 품질 관련 정량적 수치
- TIMESTAMP : 제품이 공정에 들어간 시각
- LINE : 제품이 들어간 공정 LINE 종류 ('T050304', 'T050307', 'T100304', 'T100306', 'T010306', 'T010305' 존재)
- PRODUCT_CODE : 제품의 CODE 번호 ('A_31', 'T_31', 'O_31' 존재)
- X_1 ~ X_2875 : 공정 과정에서 추출되어 비식별화된 변수

### test.csv
- PRODUCT_ID : 제품의 고유 ID
- TIMESTAMP : 제품이 공정에 들어간 시각
- LINE : 제품이 들어간 공정 LINE 종류 ('T050304', 'T050307', 'T100304', 'T100306', 'T010306', 'T010305' 존재)
- PRODUCT_CODE : 제품의 CODE 번호 ('A_31', 'T_31', 'O_31' 존재)
- X_1 ~ X_2875 : 공정 과정에서 추출되어 비식별화된 변수

### sample_submission.csv - 제출 양식
- PRODUCT_ID : 제품의 고유 ID
- Y_Class : 예측한 제품 품질 상태 - 0 : 적정 기준 미달 (부적합) / 1 : 적합 / 2 : 적정 기준 초과 (부적합)

## EDA
데이터는 2875개의 컬럼이 보안상의 이유로 비식별화 처리되어 있어 Feature Engineering에 한계가 있었습니다.
기본적으로 2000개가 넘는 컬럼 수를 최대한 줄이기 위해 **A 제품**과 데이터 특성이 비슷한 **T,O 제품** 각 2가지로 나누어 분석했습니다. (많은 null컬럼 삭제 가능.)
### 기본적인 기술통계
### 결측치 분포
### 종속 변수 분포
![Y_Class](https://github.com/user-attachments/assets/2f8f9950-a846-4cf4-a484-c73744a2b21e)
![Y_Quality_Density by Y_Class](https://github.com/user-attachments/assets/c0732bfd-5633-4fb2-bde8-5dbc6747e2d3)
### 전체 데이터 분포 확인
### 이상치 분석
트리기반 모델을 사용하기에 필요없지만 데이터에 대한 이해를 위해 진행했습니다.

### 상관관계분석
상관관계분석을 위한 결측치 처리에 KNN Imputer를 사용하기 위해 k값을 바꿔가며 평균 상관계수 분석
![평균 상관계수 그래프 (A / T,O)](https://github.com/user-attachments/assets/87e05475-e082-4e0e-80ba-a5c2eace0d80)

![히트맵(random100 A)](https://github.com/user-attachments/assets/06b53298-5e91-422e-afc0-be3e9e730d1c)
--------------------------------------
## DataPreprocessing & Experiment
### 1.PCA
**데이터 선형/비선형 파악**
![상관계수 히스토그램](https://github.com/user-attachments/assets/ca33da53-a9ac-4f0f-bf51-d97d7aab5948)

두 데이터 모두 0에 가까운 컬럼들이 월등히 많아서 데이터가 선형관계라고 보기는 어렵고 PCA를 했을 경우 모델 성능에 좋을지 알아보기 위해 데이터의 분산을 분석했습니다.

**데이터 분산 분석**
![분산 분석](https://github.com/user-attachments/assets/aa96f6d2-37b1-4a9f-9dcb-6493ab44badd)

A 데이터의 결과를 보면 29개의 컬럼으로 약 85퍼센트의 데이터 분산을 설명할 수 있으며, 소수의 주성분으로 데이터의 대부분의 분산을 설명할 수 있으므로 PCA 적용이 효과적일 것이라 봅니다. 따라서 선형 모델인 SVM도 실험에 추가해봅니다. 
그러나 T/O 데이터의 경우 78개의 컬럼으로 약 85퍼센트의 데이터 분산을 설명할 수 있으며, 소수의 주성분으로 데이터의 대부분의 분산을 설명할 수 없으므로 PCA 적용이 오히려 모델의 성능을 더 떨어트릴 것이라 예상합니다. 
### 2. 상관관계가 높은 컬럼 제거
### 3. XGBoost의 feature importance로 중요도 낮은 컬럼 제거
### 4. null 컬럼과 고윳값 컬럼 제거 (기본 전처리)
### 5. VIF값이 10 초과하는 컬럼 제거
### 실험 결과
A_31 제품 데이터의 경우 GridSearchCV를 통해 나온 평균 성능(0.7324)이나 테스트 데이터를 이용한 평가지표들을보았을 때 a_df_pca 데이터를 XGBoost 모델에 학습시키는 것이 가장 좋은 결과를 보여주고 있기에 **PCA로 차원축소한** 데이터를 가지고 모델 훈련 및 예측을 수행합니다.
T_31/O_31 제품 데이터의 경우 테스트 데이터 성능을 보았을 때 to_df_coefX와 to_df_imputed, to_df_importantFeature 데이터들 모두 0.8491의 정확도를 보이고 있습니다. 따라서 이 3가지의 데이터를 더 비교해보았을 때, ROC-AUC 값이 가장 높고, LogLoss값이 가장 낮은 것은 to_df_coefX 데이터이기에 최종적으로 **상관계수가 높은 컬럼들을 제거한** 데이터를 가지고 모델 훈련 및 예측을 수행합니다.

## Train & Predict (Parameter Tuning)
