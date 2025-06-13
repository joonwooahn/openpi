"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

### david add !!!
import openpi.transforms_david as _transforms_david
import json, pathlib # norm_stats 파일을 읽기 위해 json 모듈 임포트
from typing import Dict, Optional, Tuple # Dict는 typing 모듈에서 가져옵니다.  
import numpy as np  
import os

ACTION_CHUNK_HORIZON = 50
BATCH_SIZE = 24
MAX_TOKEN_LEN = 384
TRAIN_STEP = 50_000
# CHECKPOINT_BASE_DIR  = "./checkpoints/chunk_50"
CHECKPOINT_BASE_DIR  = "./checkpoints"
### david add !!!

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""

@dataclasses.dataclass(frozen=True) # david move!!! 이 함수를 DataCMyCustomLeRobotDataConfig위로 이동
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(max_len=model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(max_len=model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

### david add !!!
@dataclasses.dataclass(frozen=True)
class MyCustomLeRobotDataConfig(DataConfigFactory): # DataConfigFactory 상속 유지 (create_base_config 사용 위함)
    repo_id: str | None = None
    local_files_only: bool = True
    prompt_from_task: bool = False
    
    action_sequence_keys: Tuple[str, ...] = ("action",) # <--- 여기를 ("actions",)에서 ("action",)으로 수정

    repack_transforms_custom: _transforms.Group = dataclasses.field(default_factory=lambda: _transforms.Group(inputs=[], outputs=[]))
    data_transforms_custom: _transforms.Group = dataclasses.field(default_factory=lambda: _transforms.Group(inputs=[], outputs=[]))
    model_transforms_custom: _transforms.Group = dataclasses.field(default_factory=lambda: _transforms.Group(inputs=[], outputs=[]))
    
    # create 메서드를 부모의 기능을 활용하도록 수정합니다.
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 이 메서드는 self.assets (AssetsConfig)를 사용하여 norm_stats를 로드합니다.
        data_config = self.create_base_config(assets_dirs)

        # 2. 부모가 만든 기본 DataConfig에 우리의 커스텀 설정을 덮어씁니다.
        return dataclasses.replace(
            data_config,
            repack_transforms=self.repack_transforms_custom,
            data_transforms=self.data_transforms_custom,
            model_transforms=self.model_transforms_custom,
            # 기타 필요한 필드들도 여기서 다시 설정해줄 수 있습니다.
            repo_id=self.repo_id,
            action_sequence_keys=self.action_sequence_keys,
            # PI0-FAST 모델은 보통 quantile 정규화를 사용합니다.
            use_quantile_norm=(model_config.model_type == ModelType.PI0_FAST)
        )
### david add !!!

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    # checkpoint_base_dir: str = "./checkpoints"    
    # checkpoint_base_dir: str = f"/tmp/{os.environ.get('USER')}/checkpoints"  ### david add
    checkpoint_base_dir: str = CHECKPOINT_BASE_DIR
    print('================checkpoint_base_dir :', checkpoint_base_dir)
    
    # Random seed that will be used by random generators during training.
    seed: int = 42

    # Global batch size.
    # batch_size: int = 32
    batch_size: int = BATCH_SIZE ### david add !!!
    print('================batch_size: ', batch_size)
    
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    # num_workers: int = 2
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = TRAIN_STEP

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    # save_interval: int = 1000
    save_interval: int = 5000   ### david add !!!
    print('================checkpoint save_interval: ', save_interval)
    
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    # keep_period: int | None = 5000
    keep_period: int | None = 10000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

### david add !!!
######## gr1
indices_for_gr1_state = list(range(13)) + list(range(20, 31))
# print('\n\n-====================indices_for_gr1_state', indices_for_gr1_state)
# print(f"('-====================선택된 state 인덱스 개수: {len(indices_for_gr1_state)}")

indices_for_gr1_action = (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17)
# print('\n\n-====================indices_for_gr1_action', indices_for_gr1_action)
# print(f"('-====================선택된 action 인덱스 개수: {len(indices_for_gr1_action)}")

# _pi0_fast_rlwrld_gr1_model_config 정의 (이전과 동일)
_pi0_fast_rlwrld_gr1_model_config = pi0_fast.Pi0FASTConfig(
    action_dim=len(indices_for_gr1_action),  # 예: 6개의 조인트 + 6개의 그리퍼 액션
        # 0: "right_delta_x",
        # 1: "right_delta_y",
        # 2: "right_delta_z",
        # 3: "right_delta_theta_x",
        # 4: "right_delta_theta_y",
        # 5: "right_delta_theta_z",
        # 12: "right_thumb_proximal_yaw_joint_drive",
        # 13: "right_thumb_proximal_pitch_joint_drive",
        # 14: "right_index_proximal_joint_drive",
        # 15: "right_middle_proximal_joint_drive",
        # 16: "right_ring_proximal_joint_drive",
        # 17: "right_pinky_proximal_joint_drive",
    action_horizon=ACTION_CHUNK_HORIZON,
    max_token_len=MAX_TOKEN_LEN, 
)

_pi0_fast_rlwrld_gr1_model_config_for_finetune = pi0_fast.Pi0FASTConfig(
    action_dim=len(indices_for_gr1_action),  # 예: 6개의 조인트 + 6개의 그리퍼 액션\
    action_horizon=ACTION_CHUNK_HORIZON,
    max_token_len=MAX_TOKEN_LEN,
    paligemma_variant="gemma_2b_lora"
)

### allex
indices_for_allex_state = list(range(4)) + list(range(6, 13)) + list(range(20, 40)) # 11 + 20 = 31
# print('\n\n-====================indices_for_allex_state', indices_for_allex_state)
# print(f"('-====================선택된 state 인덱스 개수: {len(indices_for_allex_state)}")

indices_for_allex_action = (0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
# print('\n\n-====================indices_for_allex_action', indices_for_allex_action)
# print(f"('-====================선택된 action 인덱스 개수: {len(indices_for_allex_action)}")

_pi0_fast_rlwrld_allex_model_config = pi0_fast.Pi0FASTConfig(
    action_dim=len(indices_for_allex_action),
        # 0: "right_delta_x",
        # 1: "right_delta_y",
        # 2: "right_delta_z",
        # 3: "right_delta_theta_x",
        # 4: "right_delta_theta_y",
        # 5: "right_delta_theta_z",
        # 12: "right_Thumb_Yaw_Actuator",
        # 13: "right_Thumb_CMC_Actuator",
        # 14: "right_Thumb_MCP_Actuator",
        # 15: "right_Index_Roll_Actuator",
        # 16: "right_Index_MCP_Actuator",
        # 17: "right_Index_PIP_Actuator",
        # 18: "right_Middle_Roll_Actuator",
        # 19: "right_Middle_MCP_Actuator",
        # 20: "right_Middle_PIP_Actuator", 
        # 21: "right_Ring_Roll_Actuator",
        # 22: "right_Ring_MCP_Actuator",
        # 23: "right_Ring_PIP_Actuator",
        # 24: "right_Little_Roll_Actuator",
        # 25: "right_Little_MCP_Actuator",
        # 26: "right_Little_PIP_Actuator",
    action_horizon=ACTION_CHUNK_HORIZON,
    max_token_len=MAX_TOKEN_LEN, 
)

_pi0_fast_rlwrld_allex_model_config_for_finetune = pi0_fast.Pi0FASTConfig(
    action_dim=len(indices_for_allex_action),
    action_horizon=ACTION_CHUNK_HORIZON,
    max_token_len=MAX_TOKEN_LEN,
    paligemma_variant="gemma_2b_lora"
)

# print("\n\n================== INSTANCE max_token_len: ", _pi0_fast_rlwrld_allex_model_config.max_token_len)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
#################
### david add !!!
###########
### For gr1
        TrainConfig(
        name="pi0_fast_rlwrld_for_compute_norm_stats_gr1",
        model=_pi0_fast_rlwrld_gr1_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="gr1-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=False,             ### compute_norm_stats.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # OpenPI의 표준 방식인 AssetsConfig를 사용합니다.
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_gr1",
                asset_id="gr1-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[_transforms.RepackTransform(structure={
                'state': 'observation.state',
                
                ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from gr1-cube-dataset/meta/info.json
                'actions': 'action',
                'prompt': 'language_instruction',
                
                # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                'image': {
                    'robot0_robotview': 'observation.images.robot0_robotview',
                    'robot0_eye_in_right_hand': 'observation.images.robot0_eye_in_right_hand',
                    # 'robot0_eye_in_left_hand': 'observation.images.robot0_eye_in_left_hand'
                    'sideview': 'observation.images.sideview',
                },
            })], outputs=()),

            data_transforms_custom=_transforms.Group(inputs=[
                _transforms_david.SelectStateIndices(indices_to_keep=indices_for_gr1_state),

                _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_gr1_action),
                # DeltaActions는 통계 계산 시에는 필요 없을 수 있음 (또는 7D에 대해 적용)
            ], outputs=[]),
            model_transforms_custom=_transforms.Group(inputs=[], outputs=[]), # compute_norm_stats를 위해 비움
        ),
        num_train_steps=1,
    ),
    TrainConfig(
        name="pi0_fast_rlwrld_for_training_gr1",
        model=_pi0_fast_rlwrld_gr1_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="gr1-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=True,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # OpenPI의 표준 방식인 AssetsConfig를 사용
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_gr1",
                asset_id="gr1-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[
                
                ### 1. RepackTransform 입력 데이터 키 확인용 (lambda 대신 DebugPrintKeys 사용)
                # DebugPrintKeys(
                #     stage_message="BEFORE RepackTransform - Raw DataLoader Output",
                #     check_keys=(
                #         'observation.images.robot0_robotview', # info.json에 있는 대표적인 소스 이미지 키
                #         'observation.images.sideview',
                #         'observation.state',                   # info.json에 있는 소스 state 키
                #         'action',                              # info.json에 있는 소스 action 키
                #         'language_instruction'                 # info.json에 있는 소스 prompt 키
                #     )
                # ),

                _transforms.RepackTransform(structure={
                    'state': 'observation.state',
                    
                    ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from gr1-cube-dataset/meta/info.json
                    'actions': 'action',
                    'prompt': 'language_instruction',
                    
                    # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                    'image': {
                        'robot0_robotview': 'observation.images.robot0_robotview',
                        'robot0_eye_in_right_hand': 'observation.images.robot0_eye_in_right_hand',
                        # 'robot0_eye_in_left_hand': 'observation.images.robot0_eye_in_left_hand'
                        'sideview': 'observation.images.sideview',
                    },
                }),
                
                ### 2. RepackTransform 출력 데이터 키 및 'image' 키 내용 확인용 (lambda 대신 DebugPrintKeys 사용)
                # DebugPrintKeys(
                #     stage_message="AFTER RepackTransform - Check for 'image', 'state', etc.",
                #     check_keys=('image', 'state', 'actions', 'prompt'), # Repack 후 기대하는 키들
                #     check_image_dict=True # data['image'] 내부 구조와 각 카메라 이미지의 타입/shape 출력
                #     )

                ], 
                outputs=()),

                data_transforms_custom=_transforms.Group(inputs=[
                    _transforms_david.SelectStateIndices(indices_to_keep=indices_for_gr1_state),

                    _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_gr1_action),
                    _transforms.DeltaActions(mask=list(_transforms.make_bool_mask( 
                        6
                        -(_pi0_fast_rlwrld_gr1_model_config.action_dim - 6)
                    ))),
                ], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6
                        -(_pi0_fast_rlwrld_gr1_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "robot0_eye_in_right_hand",
                    "sideview",
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len))
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_gr1_model_config.action_horizon,
                    action_dim=_pi0_fast_rlwrld_gr1_model_config.action_dim
                ),
            ]),
        ),
        
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=TRAIN_STEP,
    ),
    TrainConfig(
        name="pi0_fast_rlwrld_for_finetune_training_gr1",
        model=_pi0_fast_rlwrld_gr1_model_config_for_finetune,
        data=MyCustomLeRobotDataConfig(
            repo_id="gr1-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=True,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # OpenPI의 표준 방식인 AssetsConfig를 사용합니다.
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_gr1",
                asset_id="gr1-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[
                _transforms.RepackTransform(structure={
                    'state': 'observation.state',
                
                    ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from gr1-cube-dataset/meta/info.json
                    'actions': 'action',
                    'prompt': 'language_instruction',
                    
                    # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                    'image': {
                        'robot0_robotview': 'observation.images.robot0_robotview',
                        'robot0_eye_in_right_hand': 'observation.images.robot0_eye_in_right_hand',
                        # 'robot0_eye_in_left_hand': 'observation.images.robot0_eye_in_left_hand'
                        'sideview': 'observation.images.sideview',
                    },
                }),], 
                outputs=()),

                data_transforms_custom=_transforms.Group(inputs=[
                    _transforms_david.SelectStateIndices(indices_to_keep=indices_for_gr1_state),

                    _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_gr1_action),
                    _transforms.DeltaActions(mask=list(_transforms.make_bool_mask( 
                        6
                        -(_pi0_fast_rlwrld_gr1_model_config.action_dim - 6)
                    ))),
                ], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6
                        -(_pi0_fast_rlwrld_gr1_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "robot0_eye_in_right_hand",
                    "sideview",
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len))
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_gr1_model_config.action_horizon,
                    action_dim=_pi0_fast_rlwrld_gr1_model_config.action_dim # 12
                ),
            ]),
        ),
        
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=TRAIN_STEP,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=_pi0_fast_rlwrld_gr1_model_config_for_finetune.action_dim, 
            action_horizon=_pi0_fast_rlwrld_gr1_model_config_for_finetune.action_horizon, 
            max_token_len=_pi0_fast_rlwrld_gr1_model_config_for_finetune.max_token_len, 
            paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(    ### only for inference
        name="pi0_fast_rlwrld_for_training_gr1_for_inference",
        model=_pi0_fast_rlwrld_gr1_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="gr1-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=False,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(),

            data_transforms_custom=_transforms.Group(inputs=[], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6
                        -(_pi0_fast_rlwrld_gr1_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "robot0_eye_in_right_hand",
                    "sideview",
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len))
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_gr1_model_config.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_gr1_model_config.action_horizon,
                    action_dim=_pi0_fast_rlwrld_gr1_model_config.action_dim
                ),
            ]),
        ),
    ),
#############
### For Allex
    TrainConfig(
        name="pi0_fast_rlwrld_for_compute_norm_stats_allex",
        model=_pi0_fast_rlwrld_allex_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="allex-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=False,             ### compute_norm_stats.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # OpenPI의 표준 방식인 AssetsConfig를 사용합니다.
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_allex",
                asset_id="allex-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[_transforms.RepackTransform(structure={
                'state': 'observation.state',
                
                ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from allex-cube-dataset/meta/info.json
                'actions': 'action',
                'prompt': 'language_instruction',
                
                # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                'image': {
                    'robot0_robotview': 'observation.images.robot0_robotview',
                    # 'robot0_eye_in_right_hand': 'observation.images.robot0_eye_in_right_hand',
                    # 'robot0_eye_in_left_hand': 'observation.images.robot0_eye_in_left_hand'
                    'sideview': 'observation.images.sideview',
                },
            })], outputs=()),

            data_transforms_custom=_transforms.Group(inputs=[
                _transforms_david.SelectStateIndices(indices_to_keep=indices_for_allex_state),

                _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_allex_action),
                # DeltaActions는 통계 계산 시에는 필요 없을 수 있음 (또는 7D에 대해 적용)
            ], outputs=[]),
            model_transforms_custom=_transforms.Group(inputs=[], outputs=[]), # compute_norm_stats를 위해 비움
        ),
        num_train_steps=1,
    ),
    TrainConfig(
        name="pi0_fast_rlwrld_for_training_allex",
        model=_pi0_fast_rlwrld_allex_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="allex-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=True,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # OpenPI의 표준 방식인 AssetsConfig를 사용합니다.
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_allex",
                asset_id="allex-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[
                _transforms.RepackTransform(structure={
                    'state': 'observation.state',
                    
                    ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from allex-cube-dataset/meta/info.json
                    'actions': 'action',
                    'prompt': 'language_instruction',
                    
                    # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                    'image': {
                        'robot0_robotview': 'observation.images.robot0_robotview',
                        'sideview': 'observation.images.sideview',
                    },
                }),], 
                outputs=()),

                data_transforms_custom=_transforms.Group(inputs=[
                    _transforms_david.SelectStateIndices(indices_to_keep=indices_for_allex_state),

                    _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_allex_action),
                    _transforms.DeltaActions(mask=list(_transforms.make_bool_mask( 
                        6,
                        -(_pi0_fast_rlwrld_allex_model_config.action_dim - 6)
                    ))),
                ], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6,
                        -(_pi0_fast_rlwrld_allex_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                    # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "sideview",
                    # "robot0_eye_in_right_hand",
                    # "robot0_eye_in_left_hand" # 만약 이 카메라를 사용하고 Repack에서 추가했다면 포함
                    # 사용하지 않거나 순서가 다른 카메라는 이 튜플에서 조정/제외합니다.
                    # 이 튜플의 길이가 image_mask의 길이가 됩니다.
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config.max_token_len))
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_allex_model_config.action_horizon,
                    action_dim=_pi0_fast_rlwrld_allex_model_config.action_dim
                ),
            ]),
        ),
        
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=TRAIN_STEP,
    ),
    TrainConfig(
        name="pi0_fast_rlwrld_for_finetune_training_allex",
        model=_pi0_fast_rlwrld_allex_model_config_for_finetune,
        data=MyCustomLeRobotDataConfig(
            repo_id="allex-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=True,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키
            
            # OpenPI의 표준 방식인 AssetsConfig를 사용합니다.
            assets=AssetsConfig(
                assets_dir="./assets/pi0_fast_rlwrld_for_compute_norm_stats_allex",
                asset_id="allex-cube-dataset"
            ),

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(inputs=[
                _transforms.RepackTransform(structure={
                    'state': 'observation.state',
                    
                    ### openpi가 기대하는 키들로 매핑 'openpi' : 'rlwrld dataset' #from allex-cube-dataset/meta/info.json
                    'actions': 'action',
                    'prompt': 'language_instruction',
                    
                    # "image" 키 아래에 카메라별 딕셔너리를 생성하도록 중첩 구조로 변경
                    'image': {
                        'robot0_robotview': 'observation.images.robot0_robotview',
                        # 'robot0_eye_in_right_hand': 'observation.images.robot0_eye_in_right_hand',
                        # 'robot0_eye_in_left_hand': 'observation.images.robot0_eye_in_left_hand'
                        'sideview': 'observation.images.sideview',
                    },
                }),], 
                outputs=()),

                data_transforms_custom=_transforms.Group(inputs=[
                    _transforms_david.SelectStateIndices(indices_to_keep=indices_for_allex_state),

                    _transforms_david.SelectAndMapRlwrldActions(selected_indices=indices_for_allex_action),
                    _transforms.DeltaActions(mask=list(_transforms.make_bool_mask( 
                        6,
                        -(_pi0_fast_rlwrld_allex_model_config.action_dim - 6)
                    ))),
                ], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6,
                        -(_pi0_fast_rlwrld_allex_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "sideview",
                    # "robot0_eye_in_right_hand",
                    # "robot0_eye_in_left_hand" # 만약 이 카메라를 사용하고 Repack에서 추가했다면 포함
                    # 사용하지 않거나 순서가 다른 카메라는 이 튜플에서 조정/제외합니다.
                    # 이 튜플의 길이가 image_mask의 길이가 됩니다.
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config_for_finetune.max_token_len)),
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config_for_finetune.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_allex_model_config_for_finetune.action_horizon,
                    action_dim=_pi0_fast_rlwrld_allex_model_config_for_finetune.action_dim # 12
                ),
            ]),
        ),
        
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=TRAIN_STEP,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=_pi0_fast_rlwrld_allex_model_config_for_finetune.action_dim, 
            action_horizon=_pi0_fast_rlwrld_allex_model_config_for_finetune.action_horizon, 
            max_token_len=_pi0_fast_rlwrld_allex_model_config_for_finetune.max_token_len, 
            paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    
    TrainConfig(
        name="pi0_fast_rlwrld_for_training_allex_for_inference",
        model=_pi0_fast_rlwrld_allex_model_config,
        data=MyCustomLeRobotDataConfig(
            repo_id="allex-cube-dataset",
            # local_files_only=True,    # 새 버젼 (2025/06/06 이후) lerobot git에서는 사용안하도록함
            prompt_from_task=False,              ### train.py 실행용
            action_sequence_keys=('action',),  # 최종적으로 OpenPI가 사용할 action 키

            # MyDatasetConfig에 정의한 커스텀 transform 인자들
            repack_transforms_custom=_transforms.Group(),

                data_transforms_custom=_transforms.Group(inputs=[], 
                outputs=[
                    _transforms.AbsoluteActions(mask=list(_transforms.make_bool_mask( 
                        6,
                        -(_pi0_fast_rlwrld_allex_model_config.action_dim - 6)
                    ))),
            ]),

            model_transforms_custom=_transforms.Group(inputs=[
                _transforms.InjectDefaultPrompt(prompt=None), 
    
                _transforms_david.ConvertImageDictTensorsToNumpyAndStack(),  # <--- 스태킹 기능이 포함된 변환 사용
                _transforms.ResizeImages(224, 224),
                _transforms_david.SqueezeImageSequenceInDict(image_dict_key="image"), # <--- 새로 추가! 출력: data['image'][cam_key]는 np.ndarray (224, 224, 3)
                _transforms_david.CreateImageMasksDict(ordered_camera_keys=(
                    # image_mask 생성 (ordered_camera_keys는 실제 사용하는 카메라와 모델이 기대하는 순서에 맞게 조정)
                    "robot0_robotview",
                    "sideview",
                    # "robot0_eye_in_right_hand",
                    # "robot0_eye_in_left_hand" # 만약 이 카메라를 사용하고 Repack에서 추가했다면 포함
                    # 사용하지 않거나 순서가 다른 카메라는 이 튜플에서 조정/제외합니다.
                    # 이 튜플의 길이가 image_mask의 길이가 됩니다.
                )),

                _transforms.TokenizeFASTInputs(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config.max_token_len))
            ], 
            outputs=[
                _transforms.ExtractFASTActions(_tokenizer.FASTTokenizer(max_len=_pi0_fast_rlwrld_allex_model_config.max_token_len),
                    action_horizon=_pi0_fast_rlwrld_allex_model_config.action_horizon,
                    action_dim=_pi0_fast_rlwrld_allex_model_config.action_dim
                ),
            ]),
        ),
    ),
### david add !!!
#################
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # local_files_only=False,  # Set to True for local-only datasets.
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                # local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]