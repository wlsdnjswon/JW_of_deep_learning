# 의미론적 분할 이진분류
>아주 큰 사이즈의 컬러이미지로 의미론적 분할 작업을 하는 기초 설명 코드입니다.

1. 제로 패딩을 넣은 뒤 512 사이즈로 데이터셋을 나눈다.
2. 512 사이즈로 나뉜 데이터셋을 가지고 U-Net을 학습한다.
3. 학습된 모델을 가지고 큰 사이즈의 이미지를 나누어 예측하고 다시 결합한다.
4. 예측된 이미지를 가지고 노이즈를 제거하고 bbox와 중심 위치를 찾아낸다.

# 의미론적 분활과 인스턴스 분할 결과 
![image.jpg1](https://github.com/wlsdnjswon/J_deep_learning/assets/71718618/b9afd063-ae67-405d-9eb8-6cca0b784e6d) |![image.jpg2](https://github.com/wlsdnjswon/J_deep_learning/assets/71718618/7a0ad5ec-b33f-4e8c-a2b1-fd29b6ed0e27)


