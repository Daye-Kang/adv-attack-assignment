"""
시각화 유틸리티
- 원본 이미지 / 적대적 이미지 / 섭동(perturbation)을 나란히 표시
- results/ 디렉토리에 PNG로 저장
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_attack(original, adversarial, true_label, orig_pred, adv_pred,
                     attack_name, dataset_name, sample_idx, eps,
                     target_label=None, save_dir="results"):
    """
    공격 결과를 시각화하고 PNG로 저장한다.
    [원본 이미지] | [적대적 이미지] | [섭동 (확대)] 3장을 나란히 배치.

    Args:
        original     : 원본 이미지 텐서 [C, H, W]
        adversarial  : 적대적 이미지 텐서 [C, H, W]
        true_label   : 실제 정답 라벨
        orig_pred    : 원본에 대한 모델 예측
        adv_pred     : 적대적 이미지에 대한 모델 예측
        attack_name  : 공격 방법 이름 (예: "fgsm_targeted")
        dataset_name : 데이터셋 이름 (예: "mnist", "cifar10")
        sample_idx   : 샘플 번호
        eps          : 사용된 epsilon 값
        target_label : targeted 공격에서 목표 라벨 (untargeted면 None)
        save_dir     : 저장 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)

    # 텐서 → numpy 변환 (C, H, W) → (H, W, C)
    orig_img = original.cpu().detach().numpy()
    adv_img = adversarial.cpu().detach().numpy()
    perturbation = adv_img - orig_img

    # 채널 순서 변환: (C, H, W) → (H, W, C)
    if orig_img.shape[0] == 1:
        # MNIST: 흑백 → squeeze해서 2D로
        orig_img = orig_img.squeeze(0)
        adv_img = adv_img.squeeze(0)
        perturbation = perturbation.squeeze(0)
        cmap = "gray"
    else:
        # CIFAR-10: RGB → transpose
        orig_img = np.transpose(orig_img, (1, 2, 0))
        adv_img = np.transpose(adv_img, (1, 2, 0))
        perturbation = np.transpose(perturbation, (1, 2, 0))
        cmap = None

    # 섭동 확대: 보기 좋게 스케일링
    perturbation_vis = perturbation - perturbation.min()
    if perturbation_vis.max() > 0:
        perturbation_vis = perturbation_vis / perturbation_vis.max()

    # 3장 나란히 그리기
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # 원본 이미지
    axes[0].imshow(orig_img, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title(f"Original\nTrue: {true_label} / Pred: {orig_pred}")
    axes[0].axis("off")

    # 적대적 이미지
    axes[1].imshow(adv_img, cmap=cmap, vmin=0, vmax=1)
    if target_label is not None:
        axes[1].set_title(f"Adversarial (eps={eps})\nPred: {adv_pred} (target: {target_label})")
    else:
        axes[1].set_title(f"Adversarial (eps={eps})\nPred: {adv_pred}")
    axes[1].axis("off")

    # 섭동 (확대)
    axes[2].imshow(perturbation_vis, cmap=cmap)
    axes[2].set_title("Perturbation\n(magnified)")
    axes[2].axis("off")

    plt.suptitle(f"{attack_name} on {dataset_name.upper()}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    # 저장
    filename = f"{dataset_name}_{attack_name}_sample{sample_idx}_eps{eps}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath