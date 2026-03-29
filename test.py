"""
test.py - 메인 실행 스크립트

실행하면 다음을 순차적으로 수행:
1. MNIST 모델 학습 (또는 저장된 가중치 로드)
2. CIFAR-10 모델 fine-tuning (또는 저장된 가중치 로드)
3. 4가지 공격을 두 데이터셋에 대해 실행
4. 공격 성공률 출력 + 다양한 eps에 대한 결과 테이블
5. 시각화 이미지를 results/에 저장
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models.mnist_model import MNISTClassifier, train_mnist
from models.cifar10_model import build_cifar10_model, train_cifar10
from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted
from utils.visualize import visualize_attack


# ============================================================
# CIFAR-10 정규화 래퍼
# 공격 함수는 [0, 1] 범위의 이미지를 받아야 하므로,
# 정규화를 모델 내부에서 처리한다.
# ============================================================
class NormalizedModel(nn.Module):
    """모델 입력 전에 ImageNet 정규화를 자동 적용하는 래퍼"""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        # (1, 3, 1, 1) 형태로 저장하여 이미지 텐서와 브로드캐스트 가능하게
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        # 모델에 넣기 전에 정규화 적용
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)


# ============================================================
# 설정값
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ATTACK_SAMPLES = 200    # 공격 평가에 사용할 샘플 수
NUM_VIS_SAMPLES = 5          # 시각화할 샘플 수
EPS_LIST = [0.05, 0.1, 0.2, 0.3]  # 보고서용 epsilon 목록

# PGD 하이퍼파라미터
PGD_STEPS_MNIST = 40
PGD_EPS_STEP_MNIST = 0.01
PGD_STEPS_CIFAR = 20
PGD_EPS_STEP_CIFAR = 0.005


# ============================================================
# 헬퍼 함수
# ============================================================
def get_target_label(true_label, num_classes=10):
    """targeted 공격용 목표 라벨 생성 (정답의 다음 클래스)"""
    return (true_label + 1) % num_classes


def evaluate_attack(model, attack_fn, data_loader, eps, device,
                    targeted=False, pgd_params=None):
    """
    공격 성공률을 계산한다.

    Args:
        model       : 공격 대상 모델
        attack_fn   : 공격 함수 (fgsm_targeted 등)
        data_loader : 테스트 데이터 로더
        eps         : 섭동 크기
        device      : 디바이스
        targeted    : targeted 공격 여부
        pgd_params  : PGD일 경우 {"k": k, "eps_step": eps_step}

    Returns:
        success_rate : 공격 성공률 (%)
        results      : 시각화용 결과 리스트
    """
    model.eval()
    success = 0
    total = 0
    vis_results = []  # 시각화용 결과 저장

    for images, labels in data_loader:
        for i in range(images.size(0)):
            x = images[i:i+1].to(device)           # (1, C, H, W)
            true_label = labels[i].item()

            # 원본 예측 확인 (이미 맞게 예측하는 샘플만 공격)
            with torch.no_grad():
                orig_pred = model(x).argmax(dim=1).item()
            if orig_pred != true_label:
                continue

            # 공격 수행
            if targeted:
                target = get_target_label(true_label)
                target_tensor = torch.tensor([target]).to(device)

                if pgd_params:
                    x_adv = attack_fn(model, x, target_tensor,
                                      pgd_params["k"], eps, pgd_params["eps_step"])
                else:
                    x_adv = attack_fn(model, x, target_tensor, eps)
            else:
                label_tensor = torch.tensor([true_label]).to(device)

                if pgd_params:
                    x_adv = attack_fn(model, x, label_tensor,
                                      pgd_params["k"], eps, pgd_params["eps_step"])
                else:
                    x_adv = attack_fn(model, x, label_tensor, eps)

            # 공격 후 예측
            with torch.no_grad():
                adv_pred = model(x_adv).argmax(dim=1).item()

            # 성공 판정
            if targeted:
                is_success = (adv_pred == target)
            else:
                is_success = (adv_pred != true_label)

            if is_success:
                success += 1

            total += 1

            # 시각화용 결과 저장 (성공한 것 우선)
            if len(vis_results) < NUM_VIS_SAMPLES and is_success:
                vis_results.append({
                    "original": x.squeeze(0).cpu(),
                    "adversarial": x_adv.squeeze(0).cpu(),
                    "true_label": true_label,
                    "orig_pred": orig_pred,
                    "adv_pred": adv_pred,
                    "target_label": target if targeted else None,
                })

            if total >= NUM_ATTACK_SAMPLES:
                break
        if total >= NUM_ATTACK_SAMPLES:
            break

    success_rate = 100.0 * success / total if total > 0 else 0.0
    return success_rate, vis_results


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("=" * 60)
    print("  Adversarial Attack Assignment")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. MNIST 모델 준비
    # ----------------------------------------------------------
    print("\n[Step 1] MNIST 모델 준비")
    mnist_model = MNISTClassifier().to(DEVICE)

    if os.path.exists("mnist_cnn.pth"):
        mnist_model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
        print("  -> 저장된 가중치 로드 완료 (mnist_cnn.pth)")
    else:
        print("  -> 학습 시작...")
        mnist_model = train_mnist(epochs=5, device=DEVICE)

    mnist_model.eval()

    # MNIST 테스트 데이터 로드
    mnist_test = datasets.MNIST(root="./data", train=False, download=True,
                                transform=transforms.ToTensor())
    mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    # ----------------------------------------------------------
    # 2. CIFAR-10 모델 준비
    # ----------------------------------------------------------
    print("\n[Step 2] CIFAR-10 모델 준비")
    cifar_base_model = build_cifar10_model(DEVICE)

    if os.path.exists("cifar10_efficientnet.pth"):
        cifar_base_model.load_state_dict(
            torch.load("cifar10_efficientnet.pth", map_location=DEVICE))
        print("  -> 저장된 가중치 로드 완료 (cifar10_efficientnet.pth)")
    else:
        print("  -> 학습 시작...")
        cifar_base_model = train_cifar10(epochs=5, device=DEVICE)

    # 정규화 래퍼로 감싸기 (공격이 [0,1] 이미지에서 동작하도록)
    cifar_model = NormalizedModel(
        cifar_base_model,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(DEVICE)
    cifar_model.eval()

    # CIFAR-10 테스트 데이터 로드 (정규화 없이 [0,1]만)
    cifar_test = datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),  # [0, 1] 범위, 정규화는 NormalizedModel이 처리
        ])
    )
    cifar_loader = DataLoader(cifar_test, batch_size=32, shuffle=False)

    # ----------------------------------------------------------
    # 3. 공격 실행
    # ----------------------------------------------------------
    # 공격 목록 정의
    attacks = [
        {
            "name": "fgsm_targeted",
            "fn": fgsm_targeted,
            "targeted": True,
            "pgd_params": None,
        },
        {
            "name": "fgsm_untargeted",
            "fn": fgsm_untargeted,
            "targeted": False,
            "pgd_params": None,
        },
        {
            "name": "pgd_targeted",
            "fn": pgd_targeted,
            "targeted": True,
            "pgd_params": None,  # 데이터셋별로 다르게 설정
        },
        {
            "name": "pgd_untargeted",
            "fn": pgd_untargeted,
            "targeted": False,
            "pgd_params": None,
        },
    ]

    datasets_config = [
        {
            "name": "mnist",
            "model": mnist_model,
            "loader": mnist_loader,
            "pgd_params": {"k": PGD_STEPS_MNIST, "eps_step": PGD_EPS_STEP_MNIST},
        },
        {
            "name": "cifar10",
            "model": cifar_model,
            "loader": cifar_loader,
            "pgd_params": {"k": PGD_STEPS_CIFAR, "eps_step": PGD_EPS_STEP_CIFAR},
        },
    ]

    # 결과 저장 (보고서 테이블용)
    all_results = {}

    for ds in datasets_config:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds['name'].upper()}")
        print(f"{'='*60}")

        for atk in attacks:
            print(f"\n  --- {atk['name']} ---")

            # PGD의 경우 데이터셋별 하이퍼파라미터 설정
            pgd_params = ds["pgd_params"] if "pgd" in atk["name"] else None

            # 각 epsilon에 대해 공격 실행
            for eps in EPS_LIST:
                success_rate, vis_results = evaluate_attack(
                    model=ds["model"],
                    attack_fn=atk["fn"],
                    data_loader=ds["loader"],
                    eps=eps,
                    device=DEVICE,
                    targeted=atk["targeted"],
                    pgd_params=pgd_params,
                )

                # 결과 저장
                key = (ds["name"], atk["name"], eps)
                all_results[key] = success_rate

                print(f"  eps={eps:.2f} -> Success Rate: {success_rate:.1f}%")

                # 각 epsilon마다 시각화 저장 (보고서: 섭동 가시성 비교)
                if vis_results:
                    for idx, res in enumerate(vis_results):
                        filepath = visualize_attack(
                            original=res["original"],
                            adversarial=res["adversarial"],
                            true_label=res["true_label"],
                            orig_pred=res["orig_pred"],
                            adv_pred=res["adv_pred"],
                            attack_name=atk["name"],
                            dataset_name=ds["name"],
                            sample_idx=idx,
                            eps=eps,
                            target_label=res["target_label"],
                        )
                        print(f"    Saved: {filepath}")

    # ----------------------------------------------------------
    # 4. 결과 요약 테이블 출력
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY TABLE")
    print(f"{'='*60}")

    for ds_name in ["mnist", "cifar10"]:
        print(f"\n  [{ds_name.upper()}]")
        header = f"  {'Attack':<22}" + "".join(f"eps={e:<8}" for e in EPS_LIST)
        print(header)
        print("  " + "-" * len(header))

        for atk in attacks:
            row = f"  {atk['name']:<22}"
            for eps in EPS_LIST:
                rate = all_results.get((ds_name, atk['name'], eps), 0.0)
                row += f"{rate:<8.1f}%"
            print(row)

    print(f"\n{'='*60}")
    print(f"  Visualizations saved to results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()