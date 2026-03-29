"""
FGSM (Fast Gradient Sign Method) 공격 구현
- Targeted: 특정 목표 클래스로 분류하게 만듦 (손실 최소화 → 마이너스 부호)
- Untargeted: 단순히 오분류를 유발 (손실 최대화 → 플러스 부호)
"""
import torch
import torch.nn as nn


def fgsm_targeted(model, x, target, eps):
    """
    Targeted FGSM 공격
    목표: 모델이 입력을 target 클래스로 예측하게 만든다.

    Args:
        model  : 신경망 (eval 모드여야 함)
        x      : 입력 이미지 텐서 (batch 가능)
        target : 원하는 (잘못된) 클래스 라벨
        eps    : 섭동 크기 (예: 0.1, 0.3)

    Returns:
        x_adv  : 적대적 이미지
    """
    # 입력 복사 (원본 보존) + 그래디언트 계산 활성화
    x_adv = x.clone().detach().requires_grad_(True)

    # Step 1: 모델 출력 (logits) 계산
    outputs = model(x_adv)

    # Step 2: target 라벨에 대한 손실 계산
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target)

    # Step 3: 입력에 대한 그래디언트 계산
    loss.backward()
    grad = x_adv.grad.data

    # Step 4: 적대적 이미지 생성
    # 마이너스 부호: target에 대한 손실을 "줄이는" 방향으로 이동
    # → 모델이 target을 더 확신하게 만듦
    x_adv = x.clone().detach() - eps * grad.sign()

    # Step 5: 유효한 이미지 범위 [0, 1]로 클램프
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def fgsm_untargeted(model, x, label, eps):
    """
    Untargeted FGSM 공격
    목표: 모델이 입력을 정답이 아닌 아무 클래스로 예측하게 만든다.

    Args:
        model : 신경망 (eval 모드여야 함)
        x     : 입력 이미지 텐서 (batch 가능)
        label : 정확한 클래스 라벨
        eps   : 섭동 크기

    Returns:
        x_adv : 적대적 이미지
    """
    # 입력 복사 + 그래디언트 계산 활성화
    x_adv = x.clone().detach().requires_grad_(True)

    # Step 1: 모델 출력 계산
    outputs = model(x_adv)

    # Step 2: 정답 라벨에 대한 손실 계산
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, label)

    # Step 3: 입력에 대한 그래디언트 계산
    loss.backward()
    grad = x_adv.grad.data

    # Step 4: 적대적 이미지 생성
    # 플러스 부호: 정답에 대한 손실을 "키우는" 방향으로 이동
    # → 모델이 정답을 덜 확신하게 만듦
    x_adv = x.clone().detach() + eps * grad.sign()

    # Step 5: 유효한 이미지 범위 [0, 1]로 클램프
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv