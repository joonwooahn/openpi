import dataclasses

import json, pathlib 
from typing import Dict, Optional, Any, Tuple, Sequence, List
import torch
import numpy as np
import jax.numpy as jnp # JAX NumPy import

# DataTransformFn 프로토콜을 따르는 디버깅용 transform 클래스
# 현재는 __call__ 메소드를 가진 dataclass로 정의
@dataclasses.dataclass(frozen=True)
class DebugPrintKeys:   # 디버깅을 위해
    stage_message: str = "DEBUG STAGE"
    # 특정 키의 존재 여부 및 타입, 내용을 더 자세히 보려면 옵션 추가 가능
    check_keys: tuple = () # 확인할 특정 키들의 튜플
    check_image_dict: bool = False # data['image'] 내부를 확인할지 여부

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n--- {self.stage_message} ---")
        print(f"Top-level keys: {list(data.keys())}")

        for key_to_check in self.check_keys:
            if key_to_check in data:
                print(f"Key '{key_to_check}': EXISTS, Type: {type(data[key_to_check])}")
            else:
                print(f"Key '{key_to_check}': NOT FOUND")

        if self.check_image_dict:
            image_data = data.get("image")
            print(f"data.get('image') type: {type(image_data)}")
            if isinstance(image_data, dict):
                print(f"Keys within data['image']: {list(image_data.keys())}")
                for cam_key, img_val in image_data.items():
                    print(f"  data['image']['{cam_key}'] type: {type(img_val)}")
                    if hasattr(img_val, 'shape'):
                         print(f"  data['image']['{cam_key}'] shape: {img_val.shape}") # type: ignore
            else:
                print(f"data['image'] is not a dict or not found.")

        print(f"--- End {self.stage_message} ---\n")
        return data


@dataclasses.dataclass(frozen=True)
class SelectStateIndices:
    """state 벡터에서 지정된 인덱스만 선택하여 새로운 state 벡터를 만듭니다."""
    indices_to_keep: List[int] # 유지하고 싶은 인덱스 목록

    def __call__(self, data: dict) -> dict:
        if 'state' in data:
            original_state = np.asarray(data['state'])
            
            # NumPy의 "fancy indexing"을 사용하여 원하는 인덱스만 선택합니다.
            # `...` (Ellipsis)는 배치 차원 등 앞의 모든 차원을 그대로 유지하라는 의미입니다.
            data['state'] = original_state[..., self.indices_to_keep]
            
            # 디버깅: print(f"DEBUG: State shape changed to {data['state'].shape}")
        return data

@dataclasses.dataclass(frozen=True)
class SelectAndMapRlwrldActions:     # 사용할 action만 추출하기 위함
    selected_indices: Tuple[int, ...]

    # 선택된 요소 중 hand처럼 절대값으로 유지해야 할 요소가 있다면,
    # 해당 요소의 인덱스를 지정할 수 있습니다. (DeltaActions에서 사용)
    # Pi0FAST의 마지막 #개 액션이 hand(절대값)라고 가정합니다.
    # 이 클래스 자체에서 델타/절대 처리를 하지는 않지만, 정보를 가질 수는 있습니다.

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "actions" not in data:
            # print("WARNING: 'actions' key not found in data for SelectAndMapActions. Available keys:", data.keys())
            return data # 또는 에러 발생

        raw_actions = data["actions"]

        if not isinstance(raw_actions, torch.Tensor):
            raw_actions = torch.as_tensor(raw_actions, dtype=torch.float32)

        indices_to_select = list(self.selected_indices)
        selected_actions = raw_actions[..., indices_to_select]

        data["actions"] = selected_actions
        return data

# 이미지 텐서를 PIL.Image.fromarray가 사용할 수 있는 NumPy 배열로 변환하는 헬퍼 함수
def _convert_single_image_tensor_to_numpy_for_pil(img_tensor: torch.Tensor) -> np.ndarray:
    if not isinstance(img_tensor, torch.Tensor):
        if isinstance(img_tensor, np.ndarray): # 이미 NumPy 배열인 경우
            return img_tensor
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img_tensor)}")

    # .detach()는 그래프에서 분리 (필요한 경우), CPU로 이동, NumPy로 변환
    np_image = img_tensor.detach().cpu()

    # 채널 순서 변경: PyTorch (C, H, W) -> PIL (H, W, C)
    # 이 변환은 개별 이미지 프레임에 적용됨 (예: (C,H,W) 또는 (H,W,C))
    if np_image.ndim == 3 and (np_image.shape[0] == 1 or np_image.shape[0] == 3): # (C, H, W) 형태 가정
        np_image = np_image.permute(1, 2, 0) # (H, W, C) 형태로 변경

    np_image = np_image.numpy() # NumPy 배열로 변환

    # PIL이 주로 기대하는 uint8 타입으로 변환
    # 입력이 float 타입이고 [0,1] 범위라면 [0,255] uint8로 스케일링
    if np_image.dtype in [np.float32, np.float64]:
        if np_image.min() >= 0.0 and np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        else: # 이미 [0,255] 범위의 float이거나 다른 범위라면, 단순 타입 캐스팅
            np_image = np_image.astype(np.uint8) # 데이터 손실 또는 클리핑 가능성 있음
    elif np_image.dtype != np.uint8: # 다른 숫자 타입이면 uint8로 캐스팅
        np_image = np_image.astype(np.uint8)

    return np_image

@dataclasses.dataclass(frozen=True)
class ConvertImageDictTensorsToNumpyAndStack:   # 이미지 타입 변환
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" not in data or not isinstance(data["image"], Dict):
            return data
        new_image_sub_dict = {}
        for cam_key, image_data_sequence in data["image"].items():
            processed_frames = []
            if isinstance(image_data_sequence, Sequence) and \
               not isinstance(image_data_sequence, str) and \
               image_data_sequence:
                if all(isinstance(img, torch.Tensor) for img in image_data_sequence):
                    processed_frames = [_convert_single_image_tensor_to_numpy_for_pil(img_tensor) for img_tensor in image_data_sequence]
                elif all(isinstance(img, np.ndarray) for img in image_data_sequence):
                    processed_frames = list(image_data_sequence)
                else:
                    new_image_sub_dict[cam_key] = image_data_sequence; continue
            elif isinstance(image_data_sequence, torch.Tensor):
                processed_frames = [_convert_single_image_tensor_to_numpy_for_pil(image_data_sequence)]
            elif isinstance(image_data_sequence, np.ndarray):
                processed_frames = [image_data_sequence]
            else:
                new_image_sub_dict[cam_key] = image_data_sequence; continue
            if processed_frames:
                try:
                    first_shape = processed_frames[0].shape
                    if not all(frame.shape == first_shape for frame in processed_frames):
                        # print(f"WARNING (ConvertImageDictTensorsToNumpyAndStack): Camera '{cam_key}' has frames with inconsistent shapes. Cannot stack. Passing as list.")
                        new_image_sub_dict[cam_key] = processed_frames; continue
                    new_image_sub_dict[cam_key] = np.stack(processed_frames, axis=0)
                except Exception as e:
                    # print(f"ERROR (ConvertImageDictTensorsToNumpyAndStack): Failed to stack frames for camera '{cam_key}'. Error: {e}. Passing as list.")
                    new_image_sub_dict[cam_key] = processed_frames
            else:
                new_image_sub_dict[cam_key] = image_data_sequence
        data['image'] = new_image_sub_dict
        return data

@dataclasses.dataclass(frozen=True)
class CreateImageMasksDict:     # 이미지 mask 가상으로 생성
    ordered_camera_keys: Tuple[str, ...]
    image_dict_key: str = "image"
    output_mask_key: str = "image_mask" # Observation.from_dict가 data["image_mask"]를 사용

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_input_dict = data.get(self.image_dict_key)
        generated_masks_dict = {}

        if not isinstance(image_input_dict, dict):
            print(f"WARNING (CreateImageMasksDict): Key '{self.image_dict_key}' not found or not a dict. "
                  f"Creating False JAX masks for all ordered_camera_keys.")
            for cam_key in self.ordered_camera_keys:
                generated_masks_dict[cam_key] = jnp.array(False, dtype=jnp.bool_) # <--- jnp 사용
        else:
            for cam_key in self.ordered_camera_keys:
                is_present_and_valid = False
                if cam_key in image_input_dict and image_input_dict[cam_key] is not None:
                    image_content = image_input_dict[cam_key]
                    if isinstance(image_content, np.ndarray) and image_content.size > 0 and image_content.shape[0] > 0:
                        is_present_and_valid = True
                    # 만약 image_content가 JAX 배열일 수도 있다면, isinstance(image_content, jnp.ndarray) 체크도 추가 가능
                
                generated_masks_dict[cam_key] = jnp.array(is_present_and_valid, dtype=jnp.bool_) # <--- jnp 사용

        data[self.output_mask_key] = generated_masks_dict
        # print(f"DEBUG (CreateImageMasksDict): Created '{self.output_mask_key}': {data[self.output_mask_key]}")
        return data
    
@dataclasses.dataclass(frozen=True)
class SqueezeImageSequenceInDict:   # 이미지 차원 맞추기
    """
    data['image'] 딕셔너리 내의 각 이미지 배열(예: (1, H, W, C))에서
    크기가 1인 첫 번째 차원을 제거합니다 (결과: (H, W, C)).
    """
    image_dict_key: str = "image"

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image_input_dict = data.get(self.image_dict_key)
        if isinstance(image_input_dict, dict):
            for cam_key, img_array in image_input_dict.items():
                # img_array는 Convert...AndStack과 ResizeImages를 거쳐 NumPy 배열일 가능성이 높음
                if isinstance(img_array, np.ndarray) and img_array.ndim > 0 and img_array.shape[0] == 1:
                    image_input_dict[cam_key] = img_array.squeeze(axis=0)
                elif isinstance(img_array, jnp.ndarray) and img_array.ndim > 0 and img_array.shape[0] == 1: # JAX 배열도 처리
                    image_input_dict[cam_key] = img_array.squeeze(axis=0) # type: ignore
        return data
