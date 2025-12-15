#!/usr/bin/env python3
"""
Quick Start Script for BLIP VQAv2 Fine-tuning
빠른 테스트와 검증을 위한 스크립트
"""

import os
import subprocess
import sys

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import torch
        import transformers
        import wandb
        from PIL import Image
        print("✅ 모든 필수 패키지가 설치되어 있습니다.")
        return True
    except ImportError as e:
        print(f"❌ 필수 패키지가 누락되었습니다: {e}")
        print("다음 명령어로 설치해주세요: pip install -r requirements.txt")
        return False

def check_data():
    """Check if preprocessed data exists"""
    train_file = "dataset/VQAv2/train.json"
    val_file = "dataset/VQAv2/val.json"
    
    if os.path.exists(train_file) and os.path.exists(val_file):
        print("✅ 전처리된 데이터가 존재합니다.")
        return True
    else:
        print("❌ 전처리된 데이터가 없습니다.")
        print("먼저 preprocess_vqav2.py를 실행해주세요.")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU 사용 가능: {gpu_name}")
            return True
        else:
            print("⚠️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
            return False
    except:
        print("⚠️ GPU 상태를 확인할 수 없습니다.")
        return False

def run_quick_test():
    """Run a quick test with small dataset"""
    print("\n🚀 빠른 테스트를 시작합니다...")
    print("소규모 데이터셋으로 1 에포크 학습을 실행합니다.")
    
    cmd = [
        "python", "blip_finetune.py",
        "--max_train_samples", "100",
        "--max_val_samples", "50", 
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "4",
        "--per_device_eval_batch_size", "4",
        "--learning_rate", "2e-5",
        "--logging_steps", "5",
        "--wandb_name", "quick-test",
        "--output_dir", "./quick-test-output"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 빠른 테스트가 성공적으로 완료되었습니다!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 테스트 실행 중 오류가 발생했습니다: {e}")
        return False

def run_single_experiment():
    """Run a single experiment with reasonable dataset size"""
    print("\n🔥 단일 실험을 시작합니다...")
    print("중간 규모 데이터셋으로 3 에포크 학습을 실행합니다.")
    
    cmd = [
        "python", "blip_finetune.py",
        "--max_train_samples", "1000",
        "--max_val_samples", "500",
        "--num_train_epochs", "3", 
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "2e-5",
        "--wandb_name", "single-experiment"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 단일 실험이 성공적으로 완료되었습니다!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실험 실행 중 오류가 발생했습니다: {e}")
        return False

def run_hyperparameter_sweep():
    """Run hyperparameter sweep"""
    print("\n🎯 하이퍼파라미터 스위프를 시작합니다...")
    print("Bayesian optimization으로 최적 하이퍼파라미터를 찾습니다.")
    
    cmd = ["python", "run_sweep.py", "--count", "10"]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ 하이퍼파라미터 스위프가 성공적으로 완료되었습니다!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 스위프 실행 중 오류가 발생했습니다: {e}")
        return False

def main():
    print("🤖 BLIP VQAv2 Fine-tuning Quick Start")
    print("=" * 50)
    
    # Check system requirements
    print("\n📋 시스템 요구사항 확인...")
    if not check_requirements():
        return
    
    if not check_data():
        return
        
    check_gpu()
    
    # Show options
    print("\n📝 실행 옵션을 선택해주세요:")
    print("1. 빠른 테스트 (100 샘플, 1 에포크)")
    print("2. 단일 실험 (1000 샘플, 3 에포크)")
    print("3. 하이퍼파라미터 스위프 (10 runs)")
    print("4. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-4): ").strip()
            
            if choice == "1":
                run_quick_test()
                break
            elif choice == "2":
                run_single_experiment()
                break
            elif choice == "3":
                run_hyperparameter_sweep()
                break
            elif choice == "4":
                print("프로그램을 종료합니다.")
                break
            else:
                print("올바른 번호(1-4)를 입력해주세요.")
                
        except KeyboardInterrupt:
            print("\n\n프로그램이 중단되었습니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 