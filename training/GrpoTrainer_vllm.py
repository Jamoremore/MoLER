import os

os.environ.pop("http_proxy", None)
os.environ.pop("no_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ftp_proxy", None)

from train_utils.prompt import MQR_PROMPT, CQE_PROMPT
import sys
import requests

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

import logging

import textwrap
from typing import Any, Callable, Optional, Union, Callable, Dict, List, Tuple

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from copy import deepcopy
import re
import ast
import shutil
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import random
from transformers.trainer import (
    logger,
    safe_globals,
    set_rng_state_for_device,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xla_available,
    ParallelMode,
)

if is_torch_xla_available():
    from transformers.trainer import xm

from concurrent.futures import ThreadPoolExecutor

from contextlib import nullcontext
from transformers.utils import is_peft_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from collections import defaultdict
from trl.models import create_reference_model
from trl.models.utils import remove_hooks, add_hooks
from train_data_classes import GRPOConfig
from vllm_client import VLLMClient
from profiling import profiling_context, profiling_decorator
import json
from unittest.mock import patch
import openai
from jinja2 import Template
from transformers import trainer_utils
import json
import jieba
import jieba.analyse
import numpy as np
from collections import Counter

if is_peft_available():
    from peft import LoraConfig, PeftConfig, get_peft_model, PeftModel

if is_wandb_available():
    import wandb

from accelerate.utils import is_deepspeed_available, set_seed, gather_object, broadcast_object_list, is_peft_model
from accelerate.utils.other import is_compiled_module

if is_deepspeed_available():
    import deepspeed

from transformers.utils.import_utils import _is_package_available

_vllm_available = _is_package_available("vllm")
_rich_available = _is_package_available("rich")


def is_vllm_available() -> bool:
    return _vllm_available


def is_rich_available() -> bool:
    return _rich_available


if is_vllm_available():
    from vllm import LLM as vllmLLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

from contextlib import contextmanager
from peft import get_peft_model_state_dict, get_peft_config


def prepare_deepspeed(model, accelerator):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def replace_last_occurrence(s, old, new: str = ""):
    # 找到最后一个 old 出现的位置
    last_index = s.rfind(old)

    # 如果找到了 old
    if last_index != -1:
        # 通过切片和拼接来替换
        return s[:last_index] + new + s[last_index + len(old):]
    else:
        # 如果没有找到 old，返回原字符串
        return s



def cal_recall(node, ans, i, process_method):
    """计算召回率"""
    url = ""
    if process_method.startswith("recall_sci"):
        url = "http://10.208.65.74:30804/service16/v1/v5"
        n = 10
        response = requests.post(
            url,
            json={"n": n, "query": node, "query_id": ans[-1]["query_id"]}
        )
    elif process_method.startswith("recall_nfc"):
        url = "http://10.208.65.74:30804/service16/v1/v2"
        n = 1000
        response = requests.post(
            url,
            json={"n": n, "query": node, "query_id": ans[-1]["query_id"]}
        )
    try:
        recall = response.json()["Recall@"+str(n)]
    except Exception as e:
        print(f"调用失败: {e}")
        recall = 0.0

    return recall




def cal_score(nodes, ans, process_methods, loss_methods, generation_num, device):
    # nodes: 模型答案
    # ans: 输入语料, """[{"role": "user", "content": "hhh"}, {"role": "assistant", "content": "hhh"}]"""

    res = [[] for _ in range(len(process_methods))]
    loss_method_mask = [[] for _ in range(len(loss_methods))]
    for i in range(len(nodes)):
        # if i == 0:
        # 	print(len(nodes)) # 16
        process_method = process_methods[i // generation_num]
        loss_method = loss_methods[i // generation_num]

        if loss_method == "self":
            loss_method_mask[i // generation_num].append(0)
        else:
            loss_method_mask[i // generation_num].append(1)

        p = nodes[i].split("</t>")[-1] if "</t>" in nodes[i] else nodes[i]
        a = ans[i // generation_num].split("</t>")[-1] if "</t>" in ans[i // generation_num] else ans[
            i // generation_num]
        if process_method.startswith("recall"):
            score = cal_recall(p, a, i, process_method)
        res[i // generation_num].append(score)
    for i in range(len(process_methods)):
        if process_methods[i] == "time":
            if all(x is False for x in res[i]):
                res[i] = [-1.0] * len(res[i])
            else:
                min_value = min(x for x in res[i] if x is not False)
                res[i] = [min_value - 1.0 if x is False else x for x in res[i]]
    res = [item for sublist in res for item in sublist]
    loss_method_mask = [item for sublist in loss_method_mask for item in sublist]
    return torch.tensor(res, device=device), torch.tensor(loss_method_mask, device=device)


class GRPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel, nn.Module] = None,
            reward_model: Callable = None,
            args: GRPOConfig = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional["PeftConfig"] = None,
            ref_adapter_name: Optional[str] = None,
            gptq: bool = False,
    ):
        # Args
        self.ref_adapter_name = ref_adapter_name
        self.gptq = gptq
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # LoRA request for the current checkpoint weights.
        # Kept as state to account for gradient accumulation
        self.lora_request = None

        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            if ref_adapter_name:
                adapter_config = model.peft_config[ref_adapter_name]
                self.ref_model = PeftModel.from_pretrained(model, adapter_config.base_model_name_or_path)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward model
        self.reward_model = reward_model

        # Reward processing class
        # if reward_processing_class is None:
        #     reward_processing_class = AutoTokenizer.from_pretrained(reward_model.config._name_or_path)
        # if reward_processing_class.pad_token_id is None:
        #     reward_processing_class.pad_token = reward_processing_class.eos_token
        # self.reward_processing_class = reward_processing_class
        # # The reward model computes the reward for the latest non-padded token in the input sequence.
        # # So it's important to set the pad token ID to the padding token ID of the processing class.
        # self.reward_model.config.pad_token_id = reward_processing_class.pad_token_id

        # Data loading and preprocessing
        if data_collator is None:

            def data_collator(features):  # No data collation is needed in GRPO
                result = defaultdict(list)
                # 遍历原始列表
                for item in features:
                    for key, value in item.items():
                        result[key].append(value)

                # 将 defaultdict 转换为普通字典（如果需要）
                result = dict(result)
                return result

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.pad_id = processing_class.pad_token_id
        self.beta = args.beta
        self.repetition_penalty = args.repetition_penalty
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p

        self.vllm_mode = 'colocate'
        self.vllm_gpu_memory_utilization = 0.4  # only applies to colocation mode
        self.vllm_tensor_parallel_size = 1  # only applies to colocation mode

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"completion_length": [], "kl": [], "reward": [], "reward_std": []}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        TEMPLATE_STR = self.processing_class.chat_template
        self.template = Template(TEMPLATE_STR)
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)
        # set_seed(args.seed, device_specific=True)
        self.use_vllm = args.use_vllm
        self.ds3_gather_for_generation = args.ds3_gather_for_generation

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, server_port=args.vllm_server_port,
                                              connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )
                self.llm = vllmLLM(
                    model=model.name_or_path,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                                 * self.vllm_tensor_parallel_size
                                 * self.args.gradient_accumulation_steps,
                    max_model_len=self.max_prompt_length * 2 + self.max_completion_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    disable_custom_all_reduce=True,
                    max_num_batched_tokens=4096,
                )
            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=1,
                pad_token_id=processing_class.pad_token_id,
            )
        self.is_peft_model = is_peft_available() and (isinstance(model, PeftModel) or is_peft_model(model))
        self.run_dir = self.args.output_dir

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if is_torch_xla_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])

        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
        if is_torch_npu_available():
            set_rng_state_for_device("NPU", torch.npu, checkpoint_rng_state, is_distributed)
        if is_torch_hpu_available():
            set_rng_state_for_device("HPU", torch.hpu, checkpoint_rng_state, is_distributed)
        if is_torch_mlu_available():
            set_rng_state_for_device("MLU", torch.mlu, checkpoint_rng_state, is_distributed)
        if is_torch_musa_available():
            set_rng_state_for_device("MUSA", torch.musa, checkpoint_rng_state, is_distributed)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
        # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @contextmanager
    def null_ref_context(self, model):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
                model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                model.set_adapter("default")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = self.accelerator.device
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        inputs_message_old = [x[-2]["content"] for x in inputs["messages"]]

        for x in inputs["messages"]:
            x[-2]["content"] = MQR_PROMPT.format(query=x[-2]["content"])
        prompts_text = [
            replace_last_occurrence(self.template.render({"messages": x[:-1] + [{"role": "assistant", "content": ""}]}),
                                    "<|im_end|>\n") for x in inputs["messages"]]
        process_methods = inputs["process_method"]
        loss_methods = inputs["loss_method"]
        # Generate completions
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process

            if self.vllm_mode == "colocate":
                # if self.guided_decoding_regex:
                # 	guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                # else:
                guided_decoding = None
                sampling_params = SamplingParams(
                    n=self.num_generations,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
                # 计算 prompt_completion_ids_stage_1

                completion_ids_1 = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids_1 = completion_ids_1[tp_slice]
                # Pad the completions, and concatenate them with the prompts
                completion_ids_1 = [torch.tensor(ids, device=device) for ids in completion_ids_1]
                completion_ids_1 = pad(completion_ids_1, padding_value=self.processing_class.pad_token_id)

                prompt_inputs = self.processing_class(
                    prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                )
                prompt_inputs = super()._prepare_inputs(prompt_inputs)

                if self.max_prompt_length is not None:
                    prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
                    prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

                prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations,
                                                                 dim=0)
                prompt_length_1 = prompt_inputs_repeated.size(1)
                prompt_inputs_repeated_1 = prompt_inputs_repeated

                # 第二部分，关键词提取
                inputs_messages_new = []
                for i in range(len(all_outputs)):
                    outputs = all_outputs[i].outputs
                    for output in outputs:
                        temp = copy.deepcopy(inputs["messages"][i])
                        temp[-2]["content"] = CQE_PROMPT.format(
                            original_query=inputs_message_old[i],
                            sub_query=output.text
                        )
                        inputs_messages_new.append(temp)
                prompts_text = [
                    replace_last_occurrence(
                        self.template.render({"messages": x[:-1] + [{"role": "assistant", "content": ""}]}),
                        "<|im_end|>\n") for x in inputs_messages_new]
                with profiling_context(self, "vLLM.generate"):
                    n_old = sampling_params.n
                    sampling_params.n = 1
                    all_outputs = self.llm.generate(prompts_text, sampling_params=sampling_params, use_tqdm=False)
                    sampling_params.n = n_old
                completion_ids_2 = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids_2 = completion_ids_2[tp_slice]
            # Pad the completions, and concatenate them with the prompts
            completion_ids_2 = [torch.tensor(ids, device=device) for ids in completion_ids_2]
            completion_ids_2 = pad(completion_ids_2, padding_value=self.processing_class.pad_token_id)

            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)

            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
            prompt_length_2 = prompt_inputs["input_ids"].size(1)

            prompt_inputs_repeated = prompt_inputs["input_ids"]

        # Get the per-token log probabilities for the completions for the model and the reference model
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits  # (B, L, V)
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids_2, skip_special_tokens=True)

        # todo: 支持多个reward加权
        rewards, process_methods_mask = cal_score(completions, [x for x in inputs["messages"]], process_methods,
                                                  loss_methods, self.num_generations, device)

        # ========================round-1========================
        prompt_completion_ids = torch.cat([prompt_inputs_repeated_1, completion_ids_1], dim=1)
        per_token_logps = []
        per_token_kl = []
        for i in range(prompt_completion_ids.size(0)):
            single_prompt_completion_ids = prompt_completion_ids[i].unsqueeze(0)
            single_per_token_logps = get_per_token_logps(model, single_prompt_completion_ids.detach().clone())
            single_per_token_logps = single_per_token_logps[:, prompt_length_1 - 1:]
            per_token_logps.append(single_per_token_logps)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_single_per_token_logps = get_per_token_logps(self.ref_model,
                                                                     single_prompt_completion_ids.detach().clone())
                else:
                    with self.null_ref_context(model):
                        ref_single_per_token_logps = get_per_token_logps(model,
                                                                         single_prompt_completion_ids.detach().clone())
            ref_single_per_token_logps = ref_single_per_token_logps[:, prompt_length_1 - 1:]
            # 计算当前样本的 KL 散度
            single_per_token_kl = torch.exp(ref_single_per_token_logps - single_per_token_logps) - (
                    ref_single_per_token_logps - single_per_token_logps) - 1
            per_token_kl.append(single_per_token_kl)
        per_token_logps = torch.cat(per_token_logps, dim=0)
        per_token_kl = torch.cat(per_token_kl, dim=0)

        # mask需要根据两轮对话做修改
        # Mask everything after the first EOS token
        is_eos = completion_ids_1 == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_1 = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # completion_mask_2 = (torch.zeros(prompt_inputs_repeated.size(), device=device)).int()

        completion_mask = completion_mask_1

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

        process_methods_mask = process_methods_mask.unsqueeze(1)
        per_token_loss = torch.where(
            process_methods_mask == 0,
            -((1 - self.beta) * per_token_loss - self.beta * per_token_kl),  # self, 私域
            -(self.beta * per_token_loss - (1 - self.beta) * per_token_kl),  # open, 私域
        )

        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        if process_methods[0].endswith("_dr"):
            # Dr.GRPO
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            # GRPO
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # ========================round-2========================
        prompt_completion_ids = torch.cat([prompt_inputs_repeated,completion_ids_2], dim=1)
        per_token_logps = []
        per_token_kl = []
        for i in range(prompt_completion_ids.size(0)):
            single_prompt_completion_ids = prompt_completion_ids[i].unsqueeze(0)
            single_per_token_logps = get_per_token_logps(model, single_prompt_completion_ids.detach().clone())
            single_per_token_logps = single_per_token_logps[:, prompt_length_2 - 1:]
            per_token_logps.append(single_per_token_logps)

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_single_per_token_logps = get_per_token_logps(self.ref_model,
                                                                     single_prompt_completion_ids.detach().clone())
                else:
                    with self.null_ref_context(model):
                        ref_single_per_token_logps = get_per_token_logps(model,
                                                                         single_prompt_completion_ids.detach().clone())
            ref_single_per_token_logps = ref_single_per_token_logps[:, prompt_length_2 - 1:]
            # 计算当前样本的 KL 散度
            single_per_token_kl = torch.exp(ref_single_per_token_logps - single_per_token_logps) - (
                    ref_single_per_token_logps - single_per_token_logps) - 1
            per_token_kl.append(single_per_token_kl)
        per_token_logps = torch.cat(per_token_logps, dim=0)
        per_token_kl = torch.cat(per_token_kl, dim=0)

        # mask需要根据两轮对话做修改
        # Mask everything after the first EOS token
        is_eos = completion_ids_2 == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask_2 = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_mask = completion_mask_2

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)

        process_methods_mask = process_methods_mask.unsqueeze(1)
        per_token_loss = torch.where(
            process_methods_mask == 0,
            -((1 - self.beta) * per_token_loss - self.beta * per_token_kl),  # self, 私域
            -(self.beta * per_token_loss - (1 - self.beta) * per_token_kl),  # open, 私域
        )

        if process_methods[0].endswith("_dr"):
            # Dr.GRPO
            loss += (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            # GRPO
            loss += ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"][-1] += completion_length
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"][-1] = (self._metrics["kl"][-1]+self.accelerator.gather_for_metrics(mean_kl).mean().item())/2

        if num_items_in_batch:
            if return_outputs:
                loss = (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                loss = loss / self.args.gradient_accumulation_steps
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / (len(val) + 1e-6) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics = {key: [] for key in self._metrics}
