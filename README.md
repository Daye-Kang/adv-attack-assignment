# Adversarial Attack Assignment

## 구조
adv-attack-assignment/
├── test.py              # 메인 (학습 → 공격 → 시각화 순차 실행)
├── requirements.txt
├── README.md
├── models/
│   ├── mnist_model.py   # Step 2
│   └── cifar10_model.py # Step 3
├── attacks/
│   ├── fgsm.py          # Step 4
│   └── pgd.py           # Step 5
├── utils/
│   └── visualize.py     # Step 6
└── results/             # 공격 결과 PNG 저장소

