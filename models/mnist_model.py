"""
MNIST 분류기
- 간단한 CNN (Conv2d x2 + FC layers)
- 목표 정확도: >= 95%
"""
import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """MNIST 이미지 분류를 위한 간단한 CNN 모델"""

    def __init__(self):
        super().__init__()
        # 특징 추출부: 합성곱 레이어 2개
        self.features = nn.Sequential(
            # Conv1: 1채널(흑백) -> 32채널, 3x3 커널
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            # Conv2: 32채널 -> 64채널, 3x3 커널
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        # 분류부: 완전연결 레이어
        self.classifier = nn.Sequential(
            nn.Flatten(),              # 64*7*7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),        # 10개 클래스 (0~9)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_mnist(epochs=5, lr=0.001, device="cpu"):
    """
    MNIST 모델을 학습하고 저장한다.

    Args:
        epochs: 학습 에폭 수
        lr: 학습률
        device: 사용할 디바이스 ("cpu" 또는 "cuda")

    Returns:
        model: 학습된 모델
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 데이터 로드 (0~1 범위로 정규화)
    transform = transforms.ToTensor()  # 자동으로 [0, 1] 범위
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # 모델, 손실함수, 옵티마이저
    model = MNISTClassifier().to(device)
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

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # 테스트 정확도 확인
    accuracy = evaluate_mnist(model, test_loader, device)
    print(f"MNIST Clean Accuracy: {accuracy:.2f}%")

    # 모델 저장
    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved to mnist_cnn.pth")

    return model


def evaluate_mnist(model, test_loader, device="cpu"):
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