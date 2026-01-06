# BLIP VQAv2 Fine-tuning with WandB

BLIP(ViT-L) 모델을 VQAv2 데이터셋으로 fine-tuning하고 WandB를 사용한 하이퍼파라미터 튜닝을 지원하는 프로젝트입니다.

## 📁 프로젝트 구조

```
2025_Samsung_AI_Challenge/
├── vqav2_dataset.py       # VQAv2 데이터셋 로더
├── blip_finetune.py       # 메인 fine-tuning 스크립트
├── sweep_config.yaml      # WandB sweep 설정
├── run_sweep.py          # Sweep 실행 스크립트
├── requirements.txt      # 필요한 패키지
├── README.md            # 사용법 설명
├── preprocess_vqav2.py  # 데이터 전처리 스크립트
└── dataset/
    └── VQAv2/
        ├── train.json   # 전처리된 학습 데이터
        └── val.json     # 전처리된 검증 데이터
```

## 🚀 설치 및 설정

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. WandB 설정
```bash
wandb login
```

## 💻 사용법

### 1. 전처리 실행
```bash
python preprocess_vqav2.py --base_dir ../dataset/VQAv2
```

### 2. 단일 실험 실행 (테스트용)
```bash
# 빠른 테스트를 위한 소규모 데이터셋
python run_sweep.py single --max_train_samples 1000 --max_val_samples 500

# 전체 데이터셋으로 단일 실험
python blip_finetune.py --num_train_epochs 3 --per_device_train_batch_size 16
```

### 3. 하이퍼파라미터 Sweep 실행
```bash
# 기본 설정으로 20개 실험 실행
python run_sweep.py --count 20

# 커스텀 프로젝트명으로 실행
python run_sweep.py --project "my-blip-experiment" --count 10
```

### 4. 커스텀 파라미터로 실험
```bash
python blip_finetune.py \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --weight_decay 0.05 \
    --wandb_name "custom-experiment"
```

## ⚙️ 주요 하이퍼파라미터

### 학습 관련
- `learning_rate`: 학습률 (기본값: 2e-5)
- `num_train_epochs`: 학습 에포크 수 (기본값: 3)
- `per_device_train_batch_size`: 배치 크기 (기본값: 16)
- `weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `warmup_ratio`: 워밍업 비율 (기본값: 0.1)

### 데이터 관련
- `train_data_path`: 전처리된 train JSON 경로 (기본값: ../dataset/VQAv2/train.json)
- `val_data_path`: 전처리된 val JSON 경로 (기본값: ../dataset/VQAv2/val.json)
- `max_train_samples`: 최대 학습 샘플 수 (테스트용)
- `max_val_samples`: 최대 검증 샘플 수 (테스트용)
- `max_length`: 최대 시퀀스 길이 (기본값: 512)

### Optuna 관련
- `best_params_path`: Optuna 결과 JSON 경로 (기본값: ../optuna_best_params_final.json)

### WandB 관련
- `wandb_project`: WandB 프로젝트명
- `wandb_name`: 실험 이름

## 📊 WandB Sweep 설정

`sweep_config.yaml`에서 다음 하이퍼파라미터들이 자동으로 튜닝됩니다:

- **Learning Rate**: 1e-6 ~ 1e-4 (log-uniform)
- **Batch Size**: [8, 16, 32]
- **Epochs**: [3, 5, 8]
- **Weight Decay**: 0.01 ~ 0.3 (uniform)
- **Warmup Ratio**: 0.05 ~ 0.3 (uniform)
- **LR Scheduler**: ["linear", "cosine", "polynomial", "constant_with_warmup"]

## 🎯 성능 모니터링

WandB를 통해 다음 메트릭들을 추적할 수 있습니다:

- **Training Loss**: 학습 손실
- **Evaluation Accuracy**: 검증 정확도
- **Learning Rate**: 실시간 학습률 변화
- **GPU Memory**: GPU 메모리 사용량

## 💾 모델 저장

- 최고 성능 모델이 자동으로 저장됩니다
- WandB Artifacts로 모델 버전 관리
- 로컬에 체크포인트 저장

## 🔧 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python blip_finetune.py --per_device_train_batch_size 8

# 그래디언트 누적 사용
python blip_finetune.py --gradient_accumulation_steps 2
```

### 빠른 테스트
```bash
# 소규모 데이터로 테스트
python blip_finetune.py --max_train_samples 100 --max_val_samples 50 --num_train_epochs 1
```

## 📈 결과 확인

1. **WandB Dashboard**: 실시간 학습 진행 상황
2. **로컬 저장**: `./blip-vqa-finetuned/` 폴더
3. **최종 평가**: 콘솔에 출력되는 최종 결과

## 🎉 완료 후

Fine-tuning이 완료되면:
1. 최고 성능 모델이 저장됩니다
2. WandB에서 모든 실험 결과를 비교할 수 있습니다
3. 저장된 모델로 inference를 실행할 수 있습니다

---

## 📝 예시 명령어

```bash
# 1. 빠른 테스트
python run_sweep.py single

# 2. 하이퍼파라미터 최적화
python run_sweep.py --count 15

# 3. 전체 데이터셋 학습
python blip_finetune.py --num_train_epochs 5

# 4. 커스텀 실험
python blip_finetune.py --learning_rate 5e-5 --wandb_name "high-lr-experiment"
``` 