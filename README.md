# 2025 Samsung AI Challenge – BLIP VQA Fine-tuning

## 프로젝트 소개 & 주요 성과
삼성 Collegiate Programming Challenge (AI 부문)에 참가하기 위해 **BLIP (Bootstrapping Language-Image Pre-training) ViT-L** 모델을 기반으로 Visual Question Answering(VQA) 성능을 끌어올린 실험 노트북과 최적 하이퍼파라미터 산출 결과를 공유합니다.  
- `BLIP_Hyperparameter_Tuning.ipynb`에서 **Optuna 10회 탐색**을 진행해 `eval_loss 0.2807`을 달성한 조합을 확보했습니다. (`optuna_best_params_final.json` 참고)  
- `BLIP_ViT_L_with_less_data.ipynb`는 **데이터 20%만 사용한 경량 파인튜닝 전략**으로 동일한 실험 흐름을 빠르게 재현할 수 있도록 구성되어 있습니다.  
- `Answersheet.ipynb`는 최종 리더보드 제출을 위한 예측 생성/검증 과정을 정리한 노트입니다.

---

## 리포지터리 구성
```
2025_Samsung_AI_Challenge/
├── BLIP_Hyperparameter_Tuning.ipynb   # Optuna 기반 탐색 실험
├── BLIP_ViT_L_with_less_data.ipynb    # 소량 데이터 실험 노트북
├── Answersheet.ipynb                  # 제출 전 응답 검증 노트
├── optuna_best_params_final.json      # 최종 하이퍼파라미터 기록
├── .gitignore / LICENSE
└── README.md
```

---

## 기술 스택
- **언어**: Python 3.10+
- **모델/프레임워크**: PyTorch, HuggingFace Transformers, BLIP ViT-L
- **실험 자동화**: Optuna (TPE Sampler), Weights & Biases(선택), CUDA 12.x
- **데이터/유틸**: pandas, numpy, tqdm, json, matplotlib (EDA 및 시각화)

---

## 데이터셋 (VQAv2 from Visual Question Answering 2.0)
아래 파일들은 VisualQA 공식 페이지(https://visualqa.org/download.html)에서 내려받아 `dataset/VQAv2/`에 배치했습니다. 현재 로컬 경로는 `C:\Users\Lenovo\Study\2025_Samsung_AI_Challenge\dataset\VQAv2` 입니다.

| 구분 | 사용 파일 | VisualQA 공식 다운로드 링크 |
|------|-----------|-----------------------------|
| Train Questions | `Train/questions/v2_OpenEnded_mscoco_train2014_questions.json` | https://visualqa.org/data2/vqa/v2_OpenEnded_mscoco_train2014_questions.zip |
| Train Annotations | `Train/annotations/v2_mscoco_train2014_annotations.json` | https://visualqa.org/data2/vqa/v2_Annotations_Train_mscoco.zip |
| Validation Questions | `Validation/questions/v2_OpenEnded_mscoco_val2014_questions.json` | https://visualqa.org/data2/vqa/v2_OpenEnded_mscoco_val2014_questions.zip |
| Validation Annotations | `Validation/annotations/v2_mscoco_val2014_annotations.json` | https://visualqa.org/data2/vqa/v2_Annotations_Val_mscoco.zip |
| MSCOCO Train Images | `Train/train2014/` | http://images.cocodataset.org/zips/train2014.zip |
| MSCOCO Val Images | `Validation/val2014/` | http://images.cocodataset.org/zips/val2014.zip |

추가로 실험 속도를 높이기 위해 시드 42 기반의 샘플 파일(`train_10000_seed42.json`, `val_1000_seed42.json`, `val_2000_seed42.json`)을 생성해 `dataset/VQAv2/` 루트에 보관하고 있습니다.

---

## 실험 워크플로
1. **데이터 전처리 & 샘플링**  
   - VisualQA 질문/정답 JSON과 MSCOCO 이미지를 다운로드 후 `dataset/VQAv2/Train`, `Validation` 폴더 구조에 정리합니다.  
   - 경량 실험을 위해 `train_10000_seed42.json` 등 부분 집합을 별도 생성합니다.

2. **하이퍼파라미터 탐색 (`BLIP_Hyperparameter_Tuning.ipynb`)**  
   - Optuna로 learning rate, batch size, epoch, weight decay, warmup 스텝을 탐색합니다.  
   - 총 10개의 trial 중 5개가 성공적으로 수렴했으며, best trial(#2)의 설정은 `optuna_best_params_final.json`으로 저장됩니다.

3. **소량 데이터 실험 (`BLIP_ViT_L_with_less_data.ipynb`)**  
   - GPU 리소스가 제한된 환경에서도 동일한 파이프라인을 실행할 수 있도록 데이터 비율/에폭을 줄인 버전을 제공합니다.

4. **앙상블 기반 추론 & 제출물 생성 (`Answersheet.ipynb`)**  
   - `dataset/open_dataset/test.csv`의 852개 문제와 `test_input_images/` 이미지를 읽어 **BLIP Teacher Forcing + OpenCLIP + BLIP 생성 기반 RapidFuzz** 점수 조합으로 최종 답안을 만듭니다.  
   - `sample_submission.csv`를 덮어쓰는 형태로 결과가 저장됩니다.

---

## Answersheet.ipynb – 정답 구성 과정
1. **모델 로딩 & 파라미터 제약 확인**  
   - Fine-tuned BLIP-VQA (`blip_finetuned_model_less_data/checkpoint-1250`), OpenCLIP ViT-L/14 (pretrained `openai`), FLAN-T5 Small을 로딩하고 총 파라미터가 3B 이하임을 검증합니다.
2. **문항 유형 분석**  
   - 테스트 CSV의 질문을 `what_is`, `why`, `what_might`, `cultural` 등으로 라벨링하여 각 유형 비율을 계산합니다.
3. **Teacher Forcing Score (TF)**  
   - 유형별 프롬프트 템플릿 3개와 가중치(0.5/0.3/0.2)를 사용해 BLIP Loss를 음수로 변환, 옵션별 적합도를 산출합니다.
4. **OpenCLIP Score (CLIP)**  
   - 질문+선지 문장을 토큰화하고 이미지-텍스트 코사인 유사도에 길이 패널티를 적용합니다.
5. **Multi-generation Score (GEN)**  
   - BLIP으로 2~3번 답변을 생성하고 RapidFuzz(`ratio/partial/token_sort`)로 각 선지와의 문자열 유사도를 측정, 가중 평균(0.4/0.3/0.3)을 사용합니다.
6. **정규화 & 질문별 가중치**  
   - 역사적인 평균/표준편차(`tf:-3.8±0.9`, `clip:0.28±0.16`, `gen:0.22±0.14`)로 Z-score→sigmoid 변환.  
   - 질문 유형마다 (TF, CLIP, GEN) 가중치를 달리 부여(예: `what_is`: 0.30/0.55/0.15, `why`: 0.70/0.20/0.10 등).
7. **TTA + Confidence Weighting**  
   - Resize, CenterCrop, RandomCrop, ColorJitter 네 가지 변환으로 이미지를 증강하고, 각 변환별 점수와 `max(score)-mean(score)` 기반 confidence로 가중합을 만듭니다.
8. **Softmax 후 최종 선택**  
   - TTA 결과를 temperature=1.2 softmax에 통과시켜 확률벡터를 만들고, argmax로 A/B/C/D 선택.  
   - 디버깅용으로 초기 5개 문항에 대해 raw score, 확률, 선택지를 콘솔에 출력합니다.
9. **제출 파일 저장**  
   - `sample_submission.csv`의 `answer` 컬럼을 앙상블 예측으로 교체해 최종 제출 파일로 사용합니다.

---

## 사용 방법
1. **환경 구성**
   ```bash
   conda create -n samsung-ai-2025 python=3.10 -y
   conda activate samsung-ai-2025
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install transformers optuna wandb datasets pillow matplotlib pandas tqdm
   ```
2. **데이터 다운로드**  
   - 위 표의 링크에서 압축 파일을 받아 `dataset/VQAv2/`에 동일한 구조로 압축을 해제합니다.
3. **노트북 실행 순서**  
   - `BLIP_Hyperparameter_Tuning.ipynb` → `BLIP_ViT_L_with_less_data.ipynb` (선택) → `Answersheet.ipynb`  
   - GPU 메모리가 부족하면 샘플 JSON 파일로 경로를 변경해 사용합니다.
4. **Optuna 결과 재사용**  
   - `optuna_best_params_final.json`을 참고해 원하는 스크립트/노트북에 동일한 세팅을 입력하면 전체 탐색 과정을 건너뛸 수 있습니다.
5. **정답 생성 (Answersheet)**

---

## 핵심 결과
- **Best eval loss**: `0.2807` (trial #2, 2025-07-22 06:02:40)
- **Best parameters**  
  ```
  learning_rate = 4.4447541666908114e-05
  batch_size    = 16
  num_epochs    = 2
  weight_decay  = 0.02014847788415866
  warmup_steps  = 800  (≈ warmup_ratio 0.40 when max_steps ≈ 2000)
  ```
- **Trials**: 10 total / 5 successful  
- **출력물**: Challenge 제출용 Answer Sheet, 손쉬운 재현을 위한 경량 노트북

---

## 향후 계획
1. **전체 코드 공개**: `blip_finetune.py`, `run_final_training_with_optimal_params.py` 등 로컬 스크립트를 정리하여 리포지터리에 추가할 예정입니다.  
2. **추가 실험**: 학습 스케줄러, 이미지 증강, multi-task 손실 적용으로 성능 향상.  
3. **서비스화**: 추론 파이프라인을 FastAPI/Gradio와 연동해 데모 배포.  
4. **데이터 확장**: VQAv2 test 단계 및 다른 멀티모달 QA 데이터셋(VQA-CP, GQA)으로 일반화 검증.

---

## 개인 기여
- VisualQA VQAv2 데이터 정리, 샘플 세트 구성, GPU 모니터링 템플릿 설계  
- Optuna 실험 설계/리포팅 및 베스트 하이퍼파라미터 도출  
- 소량 데이터 실험 노트북, 제출용 Answer Sheet, README 문서화

---

## 참고 자료
- Visual Question Answering 2.0 Dataset: https://visualqa.org/download.html  
- BLIP (Bootstrapping Language-Image Pre-training): https://arxiv.org/abs/2201.12086  
- Optuna Hyperparameter Optimization Framework: https://optuna.org/
