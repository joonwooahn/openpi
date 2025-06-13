#!/bin/bash

# ===============================================================
# 1. Slurm 배치 잡 옵션 (스크립트 최상단으로 이동)
# ===============================================================
#SBATCH --job-name=openpi-train-job
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

# srun --gpus=1 --nodes=1 --pty /bin/bash
# ===============================================================
# 2. 인자 파싱 및 유효성 검사
# ===============================================================
# 인자 개수 확인 (정확히 4개를 요구하도록 수정)
if [ "$#" -ne 4 ]; then
    echo "=========================================================================================================================================="
    echo "=== ERROR: Exactly 4 arguments are required."
    echo "=== Usage: $0 {gr1|allex} {True|False} {True|False} {True|False}"
    echo "===        (robot_type) (do_compute_norm_stats) (do_fine_tune) (fast_or_not)"
    echo "=========================================================================================================================================="
    exit 1
fi

robot_type="$1"
do_compute_norm_stats="$2"
do_fine_tune="$3"
fast_or_not="$4"

has_error=false

if [ "$robot_type" != "gr1" ] && [ "$robot_type" != "allex" ]; then
  echo "ERROR: First argument (robot_type) must be 'gr1' or 'allex'. Received: '$robot_type'"
  has_error=true
fi

if [ "$do_compute_norm_stats" != "True" ] && [ "$do_compute_norm_stats" != "False" ]; then
  echo "ERROR: Second argument (do_compute_norm_stats) must be 'True' or 'False'. Received: '$do_compute_norm_stats'"
  has_error=true
fi

if [ "$do_fine_tune" != "True" ] && [ "$do_fine_tune" != "False" ]; then
  echo "ERROR: Third argument (do_fine_tune) must be 'True' or 'False'. Received: '$do_fine_tune'"
  has_error=true
fi

# 오류 메시지 변수 오타 수정
if [ "$fast_or_not" != "True" ] && [ "$fast_or_not" != "False" ]; then
  echo "ERROR: Fourth argument (fast_or_not) must be 'True' or 'False'. Received: '$fast_or_not'"
  has_error=true
fi

do_compute_norm_stats=False
do_fine_tune=False
fast_or_not=True

if [ "$has_error" = true ]; then
  exit 1
fi

# ===============================================================
# 3. 환경 설정
# ===============================================================
echo "========================"
echo "Setting up environment..."

# Conda 초기화 및 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openpi_env
echo "✅ Conda environment 'openpi_env' activated."
echo "Python executable: $(which python)"

# 기타 필요한 환경 변수
export HF_LEROBOT_HOME=$PWD/data/rlwrld_dataset
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "Environment variables set."
echo "========================"

# # C-1. 저장 경로 변수 설정
# echo "=== 1. Setting up paths and directories ==="
# LOCAL_STORAGE_PATH="/tmp/$USER"
# LOCAL_CHECKPOINT_DIR="$LOCAL_STORAGE_PATH/checkpoints"  # 이것은 rlwrld/david/pi_0_fast/openpi/src/openpi/training/config.py의 checkpoint_base_dir: str = 와 동일한 내용이어야 함 !!!
# PERMANENT_STORAGE_DIR="/virtual_lab/rlwrld/david/pi_0_fast/openpi/checkpoints"
# echo "Temporary checkpoints will be saved to: $LOCAL_CHECKPOINT_DIR"
# echo "Final checkpoints will be copied to: $PERMANENT_STORAGE_DIR"
# mkdir -p "$LOCAL_CHECKPOINT_DIR"

# # C-2. 주기적 백업을 위한 백그라운드 프로세스 실행
# echo -e "\n=== 2. Starting periodic backup process in the background... ==="
# (
#     # 스크립트가 실행되는 동안 무한 반복
#     while true; do
#         # 1시간(3600초) 대기
#         sleep 3600
#         echo "[$(date)] --- Periodic Backup: Copying checkpoints to permanent storage... ---"
#         # rsync로 로컬 디스크의 내용을 영구 저장소로 동기화 (복사)
#         rsync -av "$LOCAL_CHECKPOINT_DIR/" "$PERMANENT_STORAGE_DIR/"
#     done
# ) & # '&'를 붙여 이 전체 while 루프를 백그라운드에서 실행
# # 백그라운드 프로세스의 PID(Process ID)를 저장
# BACKUP_PID=$!
# echo "=== Backup process started with PID: $BACKUP_PID"

# # C-3. 자동 정리를 위한 trap 설정 (백그라운드 프로세스 종료 포함)
# cleanup() {
#     echo "=== Cleaning up..."
#     echo "=== Killing background backup process (PID: $BACKUP_PID)..."
#     kill $BACKUP_PID
#     echo "===Removing temporary directory: $LOCAL_STORAGE_PATH"
#     rm -rf "$LOCAL_STORAGE_PATH"
# }
# trap cleanup EXIT


# ===============================================================
# 4. 메인 로직 실행
# ===============================================================
echo "=== Robot is $robot_type ! ==="

if [ "$fast_or_not" = "True" ]; then
    echo "=== Model type is pi0_Fast ! ==="
    
    if [ "$robot_type" = "gr1" ]; then
        # ... (gr1 + pi0-fast 로직) ...
        if [ "$do_compute_norm_stats" = "True" ]; then
            echo "--- Starting compute_norm_stats for gr1-fast ---"
            python3 scripts/compute_norm_stats.py --config-name=pi0_fast_rlwrld_for_compute_norm_stats_gr1
        fi
        if [ "$do_fine_tune" = "True" ]; then
            echo "--- Starting fine-tuning for gr1-fast ---"
            python3 scripts/train.py pi0_fast_rlwrld_for_finetune_training_gr1 \
                                    --exp-name=jw_pi_0-fast_test_rlwrld_data_finetune_gr1_cube \
                                    --overwrite 
        else
            echo "--- Starting training for gr1-fast ---"
            python3 scripts/train.py pi0_fast_rlwrld_for_training_gr1 \
                                    --exp-name=jw_pi_0-fast_test_rlwrld_data_all_gr1_cube \
                                    --overwrite
        fi

    elif [ "$robot_type" = "allex" ]; then
        # ... (allex + pi0-fast 로직) ...
        if [ "$do_compute_norm_stats" = "True" ]; then
            echo "--- Starting compute_norm_stats for allex-fast ---"
            python3 scripts/compute_norm_stats.py --config-name=pi0_fast_rlwrld_for_compute_norm_stats_allex
        fi
        if [ "$do_fine_tune" = "True" ]; then
            echo "--- Starting fine-tuning for allex ---"
            python3 scripts/train.py pi0_fast_rlwrld_for_finetune_training_allex \
                                    --exp-name=jw_pi_0-fast_test_rlwrld_data_finetune_allex_cube \
                                    --overwrite
        else
            echo "--- Starting training for allex ---"
            python3 scripts/train.py pi0_fast_rlwrld_for_training_allex \
                                    --exp-name=jw_pi_0-fast_test_rlwrld_data_all_allex_cube \
                                    --overwrite
        fi
    fi
else
    echo "=== Model type is pi0 ! (not Fast) ==="
    # ... (pi0 로직, 위와 유사하게 구성) ...
fi


# # C-4. 마지막 최종 결과물 복사
# TRAIN_EXIT_CODE=$? 
# echo "Training script finished with exit code $TRAIN_EXIT_CODE."
# # 학습이 성공적으로 끝났을 때, 마지막 체크포인트가 누락되지 않도록 한 번 더 복사
# if [ $TRAIN_EXIT_CODE -eq 0 ]; then
#     echo -e "\n=== Copying final results to permanent storage... ==="
#     rsync -av "$LOCAL_CHECKPOINT_DIR/" "$PERMANENT_STORAGE_DIR/"
#     echo "✅ Final results copied successfully."
# else
#     echo "❌ Training failed. A final sync is not performed, but periodic backups may exist."
# fi

# # 스크립트가 끝나면 trap에 의해 cleanup 함수가 자동으로 호출되어
# # 백그라운드 프로세스가 종료되고 임시 디렉토리가 정리됩니다.
# echo -e "\n=== 6. Job finished. Cleanup will be performed automatically. ==="

echo "========================================"
echo "=== Script finished for robot: $robot_type ==="
echo "========================================"