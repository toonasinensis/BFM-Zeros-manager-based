# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium
import numpy as np
import safetensors
from huggingface_hub import PyTorchModelHubMixin

from .model import FBcprModel as BaseFBcprModel
from .model import FBcprModelConfig


class FBcprModel(
    BaseFBcprModel,
    PyTorchModelHubMixin,
    library_name="bfmzero",
    tags=["facebook", "meta", "pytorch"],
    license="cc-by-nc-4.0",
    repo_url="https://github.com/facebookresearch/bfmzero",
    docs_url="https://bfmzero.metademolab.com/",
):
    def __init__(self, **kwargs):
        obs_dim = kwargs.pop("obs_dim")
        obs_space = gymnasium.spaces.Dict(
            {
                "proprio": gymnasium.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(obs_dim,),
                    dtype=kwargs.pop("obs_dtype", np.float32),
                )
            }
        )
        action_dim = kwargs.pop("action_dim")
        dict_cfg = kwargs
        # Massage the old config to the new one
        del dict_cfg["norm_obs"]
        dict_cfg["obs_normalizer"] = {"normalizers": {"proprio": {"name": "BatchNormNormalizerConfig"}}}
        dict_cfg["archi"]["actor"]["name"] = "simple"
        del dict_cfg["archi"]["actor"]["model"]
        dict_cfg["archi"]["actor"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "proprio"}

        dict_cfg["archi"]["f"]["name"] = "ForwardArchi"
        dict_cfg["archi"]["f"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "proprio"}

        dict_cfg["archi"]["b"]["name"] = "BackwardArchi"
        dict_cfg["archi"]["b"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "proprio"}

        dict_cfg["archi"]["critic"]["name"] = "ForwardArchi"
        dict_cfg["archi"]["critic"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "proprio"}

        dict_cfg["archi"]["discriminator"]["name"] = "DiscriminatorArchi"
        dict_cfg["archi"]["discriminator"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "proprio"}
        cfg = FBcprModelConfig(**dict_cfg)
        super().__init__(obs_space, action_dim, cfg)

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        # Load up parameters but remap parameters to the new file

        # Load state dict
        state_dict = safetensors.torch.load_file(model_file, device=map_location)

        state_dict["_obs_normalizer._normalizers.proprio._normalizer.running_mean"] = state_dict["_obs_normalizer.running_mean"]
        state_dict["_obs_normalizer._normalizers.proprio._normalizer.running_var"] = state_dict["_obs_normalizer.running_var"]
        state_dict["_obs_normalizer._normalizers.proprio._normalizer.num_batches_tracked"] = state_dict[
            "_obs_normalizer.num_batches_tracked"
        ]
        del state_dict["_obs_normalizer.running_mean"]
        del state_dict["_obs_normalizer.running_var"]
        del state_dict["_obs_normalizer.num_batches_tracked"]

        model.load_state_dict(state_dict, strict=strict)

        return model
