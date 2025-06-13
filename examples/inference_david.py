import dataclasses

import jax

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.policies import policy_config
import pathlib # pathlib 추가

print("--- inference !!! ")


# 1. 설정 로드
config_name = "pi0_fast_rlwrld_for_training_gr1" # 학습에 사용한 config 이름
config = _config.get_config(config_name)
print("+++ config: ", config)
print("+++ robot_name: ", config.data.repo_id.split('-')[0])

# --- 이미지 키 추출 로직 ---
repack_transforms_custom = config.data.repack_transforms_custom
# camera_keys = list(repack_transforms_custom.inputs[0].structure.get('image').keys())
camera_keys = [key + 'agentview' for key in list(config.data.repack_transforms_custom.inputs[0].structure.get('image').keys())]
# 결과 출력
print("+++ image:  추출된 카메라 키 리스트:", camera_keys)


# 2. 실제 체크포인트 '단계(step)' 디렉토리 경로 설정
#    YOUR_EXP_NAME_HERE 를 학습 시 사용한 exp_name으로 변경 (예: "jw_pi_0-fast_test_rlwrld_data_all_gr1_cube")
#    YOUR_STEP_NUMBER_HERE 를 로드하려는 학습 스텝으로 변경 (예: "29999")
exp_name = "jw_pi_0-fast_test_rlwrld_data_all_gr1_cube" # 예시, 실제 사용한 값으로 변경
step_number = "29999" # 예시, 실제 사용한 값으로 변경

# OpenPI 기본 체크포인트 경로 구조를 따름
base_checkpoint_dir = pathlib.Path("../checkpoints") # 스크립트 실행 위치 기준
# 또는 절대 경로: pathlib.Path("/virtual_lab/rlwrld/david/pi_0/openpi/checkpoints")

# 최종적으로 파라미터가 저장된 '스텝' 폴더까지의 경로
# 이 폴더 안에 'params' 하위 폴더가 있거나, 혹은 이 폴더 자체에 파라미터 파일들이 있어야 함
actual_step_dir = base_checkpoint_dir / config_name / exp_name / step_number

print(f"Attempting to load checkpoint from step directory: {actual_step_dir}")

key = jax.random.key(0)
restored_params = None

# 3. 파라미터 로드 시도
#    먼저 'params' 하위 폴더가 있는지 시도
try:
    print(f"Trying to load from: {actual_step_dir / 'params'}")
    restored_params = _model.restore_params(actual_step_dir / "params")
    print("Successfully loaded params from 'params' subdirectory.")
except FileNotFoundError:
    print(f"Warning: 'params' subdirectory not found in {actual_step_dir}.")
    # 'params' 하위 폴더가 없다면, 해당 스텝 디렉토리에서 직접 로드 시도
    try:
        print(f"Trying to load directly from: {actual_step_dir}")
        restored_params = _model.restore_params(actual_step_dir)
        print("Successfully loaded params directly from step directory.")
    except FileNotFoundError as e:
        print(f"Error: Model params not found at {actual_step_dir} either.")
        raise e # 원래 FileNotFoundError를 다시 발생시켜서 중단

if restored_params is None:
    raise SystemExit("Failed to load model parameters.")


# --- 2. norm_stats.json 파일 경로 정의 ---
# compute_norm_stats.py가 생성한 파일의 실제 경로로 수정해야 합니다.
actual_step_dir_assets = actual_step_dir/'assets/gr1-cube-dataset/norm_stats.json'

# --- 3. config 객체 업데이트 ---
# config.data가 MyCustomLeRobotDataConfig 인스턴스인지 확인
if isinstance(config.data, _config.MyCustomLeRobotDataConfig):
    # dataclasses.replace를 사용하여 norm_stats_file_to_load 필드가 설정된 새 data factory 인스턴스를 생성
    new_data_factory = dataclasses.replace(
        config.data,
        norm_stats_file_to_load=actual_step_dir_assets
    )
    # 기존 config에서 data 부분만 새 factory로 교체한 새 config 객체를 생성
    config = dataclasses.replace(config, data=new_data_factory)
else:
    print("Warning: The loaded config's data factory is not MyCustomLeRobotDataConfig. Norm stats might not be loaded.")

# 4. 모델 생성 및 테스트
model = config.model.load(restored_params)
# We can create fake observations and actions to test the model.
obs, act = config.model.fake_obs(), config.model.fake_act()

print("------------ obs: ", obs)
# base_0_rgb, wrist_0_rgb, base_0_rgb, tokenized_

# prompt_mask, token_ar_mask, token_loss_mask
print("------------ act: ", act)
# Sample actions from the model.
loss = model.compute_loss(key, obs, act)
print("Loss shape:", loss.shape)

#####################################################################
policy = policy_config.create_trained_policy(config, actual_step_dir)

# Run inference on a dummy example.
example = {
    'state': obs.state[0],
    'prompt': "Lift the cube from the left stand and place it on the right stand.",
    'image': {
        'robot0_robotview': obs.images['base_0_rgb'][0],
        'robot0_eye_in_right_hand': obs.images['wrist_0_rgb'][0],
        'sideview': obs.images['base_1_rgb'][0],
    }
}
action_chunk = policy.infer(example)["actions"]
print("+++ action_chunk", action_chunk)