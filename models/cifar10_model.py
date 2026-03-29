"""
CIFAR-10 분류기
- torchvision pretrained EfficientNet_B0 사용 (ImageNet 가중치)
- 마지막 분류 레이어를 10클래스로 교체 후 fine-tuning
- 목표 정확도: >= 80%
- 출처: https://pytorch.org/vision/stable/models/efficientnet.html
- 선택 이유: ResNet 대비 적은 파라미터(5.3M)로 높은 효율성 달성
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def build_cifar10_model(device="cpu"):
    """
    ImageNet pretrained EfficientNet_B0을 CIFAR-10용으로 수정한다.
    - 마지막 분류 레이어: 1280 -> 10 (CIFAR-10 클래스 수)

    Args:
        device: 사용할 디바이스

    Returns:
        model: 수정된 EfficientNet_B0 모델
    """
    # ImageNet pretrained 가중치 로드
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # 마지막 분류 레이어를 CIFAR-10용 (10클래스)으로 교체
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)

    return model.to(device)


def train_cifar10(epochs=5, lr=0.001, device="cpu"):
    """
    CIFAR-10 모델을 fine-tuning하고 저장한다.

    Args:
        epochs: 학습 에폭 수
        lr: 학습률
        device: 사용할 디바이스

    Returns:
        model: 학습된 모델
    """
    # 데이터 전처리
    # - EfficientNet_B0은 ImageNet 기준 정규화를 기대함
    # - CIFAR-10은 32x32이므로 224x224로 리사이즈
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # 학습 시 데이터 증강
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 평균
                             std=[0.229, 0.224, 0.225]),    # ImageNet 표준편차
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 데이터 로드
    train_data = datasets.CIFAR10(root="./data", train=True, download=True,
                                  transform=transform_train)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True,
                                 transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 모델 준비
    model = build_cifar10_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 진행 상황 출력 (200 배치마다)
            if (batch_idx + 1) % 200 == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{len(train_loader)}")

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # 테스트 정확도 확인
    accuracy = evaluate_cifar10(model, test_loader, device)
    print(f"CIFAR-10 Clean Accuracy: {accuracy:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "cifar10_efficientnet.pth")
    print("Model saved to cifar10_efficientnet.pth")

    return model


def evaluate_cifar10(model, test_loader, device="cpu"):
    """테스트 데이터에 대한 정확도를 계산한다."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total