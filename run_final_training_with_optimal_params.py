#!/usr/bin/env python3
"""
🏆 BLIP 최종 학습 - Optuna 최적 파라미터 자동 적용

이 스크립트는 optuna_best_params_final.json에서 최적 하이퍼파라미터를 
자동으로 읽어와서 BLIP 모델의 최종 학습을 실행합니다.

사용법:
    python run_final_training_with_optimal_params.py [옵션]

옵션:
    --test          테스트 학습 (빠른 검증용, 5K 샘플)
    --full          전체 학습 (443K 샘플, 기본값)
    --samples N     사용할 샘플 수 지정
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime


def load_optimal_params(json_file="optuna_best_params_final.json"):
    """Optuna 최적화 결과에서 하이퍼파라미터 로드"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        best_params = data['best_params']
        
        print("🎯 Optuna 최적 파라미터 로드 완료!")
        print(f"📊 최고 성능 eval_loss: {data['best_eval_loss']:.4f}")
        print(f"🏆 최적 Trial 번호: {data['best_trial_number']}")
        print(f"⏰ 최적화 완료 시간: {data['optimization_time']}")
        print(f"📈 성공한 Trial: {data['successful_trials']}/{data['total_trials']}")
        
        print("\n🔧 적용할 하이퍼파라미터:")
        for key, value in best_params.items():
            if key == 'learning_rate':
                print(f"  - {key}: {value:.2e}")
            elif key == 'weight_decay':
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")
                
        # warmup_ratio 계산
        warmup_ratio = best_params['warmup_steps'] / 2000
        best_params['warmup_ratio'] = warmup_ratio
        
        print(f"  - warmup_ratio: {warmup_ratio:.3f} (warmup_steps {best_params['warmup_steps']}에서 계산)")
        
        return best_params, data
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {json_file}")
        print("📋 먼저 하이퍼파라미터 튜닝을 완료하세요.")
        return None, None
    except Exception as e:
        print(f"❌ 파라미터 로드 오류: {e}")
        return None, None


def run_training(optimal_params, train_samples=None, val_samples=None, test_mode=False):
    """최적 파라미터로 BLIP 모델 학습 실행"""
    
    # 학습 모드 설정
    if test_mode:
        train_samples = train_samples or 5000
        val_samples = val_samples or 2000
        output_dir = "./blip_test_optimal_model"
        print("🧪 테스트 학습 모드")
        print(f"📊 학습 샘플: {train_samples:,}개")
        print(f"📊 검증 샘플: {val_samples:,}개")
        print("⏰ 예상 시간: 15-30분")
    else:
        output_dir = "./blip_final_optimal_model"
        print("🏆 전체 데이터셋 최종 학습 모드")
        if train_samples:
            print(f"📊 학습 샘플: {train_samples:,}개")
        else:
            print("📊 학습 샘플: 전체 (~443K개)")
        print("⏰ 예상 시간: 3-6시간")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습 명령어 구성
    cmd = [
        sys.executable,
        "blip_finetune.py",
        "--train_file", "../dataset/VQAv2/train.json",
        "--val_file", "../dataset/VQAv2/val.json",
        "--per_device_train_batch_size", str(optimal_params['batch_size']),
        "--per_device_eval_batch_size", str(optimal_params['batch_size']),
        "--learning_rate", str(optimal_params['learning_rate']),
        "--weight_decay", str(optimal_params['weight_decay']),
        "--warmup_ratio", str(optimal_params['warmup_ratio']),
        "--num_train_epochs", str(optimal_params['num_epochs']),
        "--output_dir", output_dir,
        "--eval_strategy", "epoch",
        "--save_strategy", "epoch",
        "--load_best_model_at_end", "true",
        "--logging_steps", "100",
        "--save_total_limit", "3",
        "--dataloader_num_workers", "4",
        "--remove_unused_columns", "false",
        "--report_to", "none"
    ]
    
    # 샘플 수 제한 추가
    if train_samples:
        cmd.extend(["--max_train_samples", str(train_samples)])
    if val_samples:
        cmd.extend(["--max_val_samples", str(val_samples)])
    
    print(f"\n🚀 실행 명령어:")
    print(" ".join(cmd))
    print("\n" + "=" * 70)
    
    # 학습 실행
    try:
        start_time = datetime.now()
        print(f"⏰ 학습 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # WandB 비활성화
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        
        result = subprocess.run(
            cmd,
            cwd="./BLIP_ViT_L_Finetuning",
            env=env,
            timeout=36000
        )
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            print(f"\n🎉 학습 완료!")
            print(f"⏰ 총 학습 시간: {runtime//3600:.0f}시간 {(runtime%3600)//60:.0f}분")
            print(f"📁 모델 저장 위치: {output_dir}")
            print("\n🏆 학습 완료! 이제 모델을 사용할 수 있습니다.")
            return True
            
        else:
            print(f"\n❌ 학습 실패")
            print(f"Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ 학습 시간 초과 (10시간)")
        return False
        
    except Exception as e:
        print(f"\n❌ 학습 실행 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BLIP 최종 학습 - Optuna 최적 파라미터 자동 적용")
    parser.add_argument("--test", action="store_true", help="테스트 학습 모드 (빠른 검증용)")
    parser.add_argument("--full", action="store_true", help="전체 학습 모드 (기본값)")
    parser.add_argument("--samples", type=int, help="사용할 학습 샘플 수")
    parser.add_argument("--val_samples", type=int, help="사용할 검증 샘플 수")
    
    args = parser.parse_args()
    
    print("🏆 BLIP 최종 학습 - Optuna 최적 파라미터 자동 적용!")
    print("=" * 70)
    
    # 최적 파라미터 로드
    optimal_params, optimization_data = load_optimal_params()
    
    if optimal_params is None:
        print("\n❌ 최적 파라미터를 로드할 수 없습니다.")
        sys.exit(1)
    
    # 학습 모드 결정
    test_mode = args.test or (not args.full and args.samples and args.samples < 50000)
    
    if not test_mode and not args.full:
        # 기본값은 전체 학습이지만 확인 요청
        confirm = input("\n전체 데이터셋으로 최종 학습을 시작하시겠습니까? (yes/no): ")
        if confirm.lower() not in ['yes', 'y', '네', 'ㅇ']:
            print("⏸️ 학습이 취소되었습니다.")
            print("💡 테스트 학습을 원하시면 --test 옵션을 사용하세요.")
            sys.exit(0)
    
    # 학습 실행
    success = run_training(
        optimal_params=optimal_params,
        train_samples=args.samples,
        val_samples=args.val_samples,
        test_mode=test_mode
    )
    
    if success:
        print("\n🎉 최종 학습 성공!")
        print("🏆 삼성 AI 챌린지 제출 준비 완료!")
    else:
        print("\n❌ 학습 실패")
        sys.exit(1)


if __name__ == "__main__":
    main() 