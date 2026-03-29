"""
PGD (Projected Gradient Descent) 공격 구현
- FGSM의 반복(iterative) 버전
- 매 스텝마다 eps_step만큼 이동 후, eps-ball로 투영(projection)
- FGSM보다 강력한 공격 (더 정밀하게 적대적 예제를 찾음)
"""
import torch
import torch.nn as nn


def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    Targeted PGD 공격
    목표: k번 반복하며 모델이 target 클래스로 예측하게 만든다.

    Args:
        model    : 신경망 (eval 모드여야 함)
        x        : 입력 이미지 텐서
        target   : 원하는 (잘못된) 클래스 라벨
        k        : 반복 횟수 (예: 10, 40)
        eps      : 총 섭동 허용량 (budget)
        eps_step : 반복당 스텝 크기

    Returns:
        x_adv    : 적대적 이미지
    """
    # Step 1: 깨끗한 입력에서 시작
    x_adv = x.clone().detach()

    # Step 2: k번 반복
    for i in range(k):
        # 그래디언트 계산을 위해 requires_grad 활성화
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # (a) FGSM 한 스텝 적용
        outputs = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward()
        grad = x_adv.grad.data

        # 마이너스 부호: target 방향으로 이동 (손실 최소화)
        x_adv = x_adv.clone().detach() - eps_step * grad.sign()

        # (b) eps-ball로 투영: 원본 입력에서 eps 이상 벗어나지 못하게 제한
        x_adv = torch.clamp(x_adv, x - eps, x + eps)

        # (c) 유효한 이미지 범위 [0, 1]로 클램프
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv


def pgd_untargeted(model, x, label, k, eps, eps_step):
    """
    Untargeted PGD 공격
    목표: k번 반복하며 모델이 정답이 아닌 아무 클래스로 예측하게 만든다.

    Args:
        model    : 신경망 (eval 모드여야 함)
        x        : 입력 이미지 텐서
        label    : 정확한 클래스 라벨
        k        : 반복 횟수
        eps      : 총 섭동 허용량
        eps_step : 반복당 스텝 크기

    Returns:
        x_adv    : 적대적 이미지
    """
    # Step 1: 깨끗한 입력에서 시작
    x_adv = x.clone().detach()

    # Step 2: k번 반복
    for i in range(k):
        # 그래디언트 계산을 위해 requires_grad 활성화
        x_adv = x_adv.clone().detach().requires_grad_(True)

        # (a) FGSM 한 스텝 적용
        outputs = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, label)
        loss.backward()
        grad = x_adv.grad.data

        # 플러스 부호: 정답에서 멀어지는 방향으로 이동 (손실 최대화)
        x_adv = x_adv.clone().detach() + eps_step * grad.sign()

        # (b) eps-ball로 투영
        x_adv = torch.clamp(x_adv, x - eps, x + eps)

        # (c) 유효한 이미지 범위 [0, 1]로 클램프
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv