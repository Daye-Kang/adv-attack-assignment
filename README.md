# Adversarial Attack Assignment

신뢰할 수 있는 인공지능 과제 #1 - 신경망에 대한 적대적 공격 구현

## 실행 방법

```
pip install -r requirements.txt
python test.py
```

test.py를 실행하면 모델 학습, 공격 수행, 시각화 저장이 순서대로 진행됩니다.
학습된 가중치 파일(mnist_cnn.pth, cifar10_efficientnet.pth)이 이미 있으면 학습을 건너뛰고 바로 공격을 실행합니다. GPU가 있으면 자동으로 사용합니다.

## 모델 선택

**MNIST** - 간단한 CNN을 직접 구현했습니다.
- Conv2d(1->32) -> ReLU -> MaxPool -> Conv2d(32->64) -> ReLU -> MaxPool -> FC(128) -> FC(10)
- Clean accuracy: 99.14%

**CIFAR-10** - torchvision의 pretrained EfficientNet_B0을 fine-tuning해서 사용했습니다.
- ResNet18 대신 EfficientNet_B0을 선택한 이유는 파라미터 수가 절반(5.3M vs 11.7M)이면서도 동등 이상의 성능을 내기 때문입니다.
- 마지막 분류 레이어를 10클래스로 교체하고 5에폭 fine-tuning 진행
- Clean accuracy: 90.69%
- 출처: https://pytorch.org/vision/stable/models/efficientnet.html

## 공격 방법

4가지 공격을 구현했고, 각각 eps ∈ {0.05, 0.1, 0.2, 0.3}에 대해 200개 샘플로 성공률을 측정했습니다.

- **FGSM Targeted** - gradient sign의 반대 방향으로 한 스텝 이동하여 target 클래스로 유도
- **FGSM Untargeted** - gradient sign 방향으로 한 스텝 이동하여 오분류 유도
- **PGD Targeted** - FGSM targeted를 k번 반복하며 매 스텝 eps-ball로 projection
- **PGD Untargeted** - FGSM untargeted를 k번 반복하며 매 스텝 eps-ball로 projection

PGD 하이퍼파라미터: MNIST(k=40, eps_step=0.01), CIFAR-10(k=20, eps_step=0.005)

## 파일 구조

```
├── test.py                 # 메인 실행 (학습 -> 공격 -> 시각화)
├── models/
│   ├── mnist_model.py      # MNIST CNN 구현 및 학습
│   └── cifar10_model.py    # EfficientNet_B0 로드 및 fine-tuning
├── attacks/
│   ├── fgsm.py             # FGSM targeted, untargeted
│   └── pgd.py              # PGD targeted, untargeted
├── utils/
│   └── visualize.py        # 원본/적대적/섭동 이미지 시각화
├── results/                # 공격 결과 PNG 파일
├── report.pdf              # 분석 보고서
└── requirements.txt
```