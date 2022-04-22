# [부스트캠프] level-02 데이터제작 팀랩업리포트

## **프로젝트 개요**

[https://lh3.googleusercontent.com/_HxOq7LLOsgJXZ64Rmpx3ifCH9UsgyIsFgfTjyzf0Lxmv7UZsZdSs_08AF1LUx1DYDYC9pvcodVRyDUwyZxxJqskTtJ6DWfmE_1KMCj_p8WsawrHLSGNlNr4UUqaPqgo42B9dy8I](https://lh3.googleusercontent.com/_HxOq7LLOsgJXZ64Rmpx3ifCH9UsgyIsFgfTjyzf0Lxmv7UZsZdSs_08AF1LUx1DYDYC9pvcodVRyDUwyZxxJqskTtJ6DWfmE_1KMCj_p8WsawrHLSGNlNr4UUqaPqgo42B9dy8I)

- 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있다.
    
    이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나이다. 
    
- OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기(Serializer) 등의 모듈로 이루어져 있습니다.
    
    이번 ‘글자검출대회’에서는 모델 학습 시 데이터 중요도 확인을 위해 **학습 모델은 고정된 상태**로 대회를 진행했다.
    

## **프로젝트 팀 구성 및 역할**

- **김승현**: data annotation
- **노창현**: data annotation
- **최홍록**: data annotation
- **최진아**: data investigation
- **최용원**: data augmentation

## **프로젝트 수행 절차 및 방법**

[https://lh6.googleusercontent.com/avD2mgeMO5GTDTcUlXaoJrtxhgktNXdOmMEv2Kx3OU-G-6HsWfCS08XVwbiN0GAxRwmZ-ScHkS-GZqc9VQHWaCsg2JWirYeV4TRxMRo-OaV25eHKCXTV5AEUNh3x4PcjkUd6-P5U](https://lh6.googleusercontent.com/avD2mgeMO5GTDTcUlXaoJrtxhgktNXdOmMEv2Kx3OU-G-6HsWfCS08XVwbiN0GAxRwmZ-ScHkS-GZqc9VQHWaCsg2JWirYeV4TRxMRo-OaV25eHKCXTV5AEUNh3x4PcjkUd6-P5U)

## **프로젝트 수행 결과**

1. **학습 데이터** 
    - ICDAR 2017 / 2019
    - 야외 실제 촬영 한글 이미지([링크](https://aihub.or.kr/aidata/33985/download))
    
2. **모델**(**EAST: An Efficient and Accurate Scene Text Detector**)
    - [https://arxiv.org/abs/1704.03155](https://arxiv.org/abs/1704.03155)
    - 데이터 제작 실습을 위해 **모델은 고정**하여 사용

[https://lh4.googleusercontent.com/0Z0FhiqVN8He5J74Ixoxj3INew3FPJdEhhQx1LkibKnn7O0YC6ePMTLSHRdw2hYG0vTm20BAvGG71mLDlRkN_AiV7SpEyVK1tmZoV_q4iO8G0DssUnr4x816rlKtVpJ6DOHKq1A3](https://lh4.googleusercontent.com/0Z0FhiqVN8He5J74Ixoxj3INew3FPJdEhhQx1LkibKnn7O0YC6ePMTLSHRdw2hYG0vTm20BAvGG71mLDlRkN_AiV7SpEyVK1tmZoV_q4iO8G0DssUnr4x816rlKtVpJ6DOHKq1A3)

1. **Augmentation**
    - **horizon Flip:** f1 score, recall, precision 낮아짐
        
        → 데이터 셋이 정방향으로 많이 구성됨으로 효과가 없었던 것으로 추정  
        
    - **vertical Flip:** f1 score, recall, precision 낮아짐
        
        → 데이터 셋이 정방향으로 많이 구성됨으로 효과가 없었던 것으로 추정  
        
    - **Hue:** f1 score, recall, precision 낮아짐
        
        → 전체 밝기가 밝거나 어두워져 글자와 배경을 구분하지 못해 효과가 없던 것으로 추정  
        
    - **saturation:** f1 score, recall, precision 낮아짐
        
        → 채도가 밝거나 어두워져 글자와 배경을 구분하지 못해 효과가 없던 것으로 추정
        
    - **Sharpen:** f1 score, recall, 낮아짐 precision은 높아짐
        
        → 이미지가 선명하게 됨으로써 precision이 증가한 것으로 추정 하지만 overfitting이 발생해 recall이 낮은 것으로 추정
        
    - **blur:** f1 score, recall, precision 낮아짐
        
        → OCR Data에서는 굵은 선들이 많기 때문에 blur가 오히려 글자 영역을 더 넓게 잡아 역효과가 나왔던 것으로 추정
        

[실험 결과](https://www.notion.so/ef02055f80574065b50ff0e817a2f49a)

## **Leader Board(**Final)

[https://lh3.googleusercontent.com/uU1Yy68QUVIweeCYQzd7sAh16s-9P6VOUXRqfXD3LkXs63GQW1VAkL_IOwyaMVawJrM9cu58X6Xdy2pxl7zH7FzfaSasfZyDbI1tOavH_CDkWE9EsCvwZz13UQn9F5CBxGPZKaTt](https://lh3.googleusercontent.com/uU1Yy68QUVIweeCYQzd7sAh16s-9P6VOUXRqfXD3LkXs63GQW1VAkL_IOwyaMVawJrM9cu58X6Xdy2pxl7zH7FzfaSasfZyDbI1tOavH_CDkWE9EsCvwZz13UQn9F5CBxGPZKaTt)

## 자체 평가 의견

- **잘한 점**
    - 추가로 데이터를 수집하여 학습 형태에 맞게 annotation을 진행하였다.
    - 다양한 augmentation을 추가하여 성능 향상을 할 수 있었다.
    - 외부 데이터를 사용하여 모델의 성능을 높일 수 있다는 점을 알게 되었다.
    - 데이터의 중요성을 깨달았다.
- **시도했으나 잘 되지 않은 점들**
    - 데이터를 추가하였으나 geometry정보 부재로 인해 성능이 좋게 나오지 않았다.