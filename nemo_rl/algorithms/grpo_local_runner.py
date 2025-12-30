# no copy write yet will add later 

"""Local runner for GRPO that can run without real GPUs using mock implementations.

This module provides utilities to run GRPO training locally for development/testing
purposes using mock Trainer and Sampler implementations that work on CPU.

The module supports two backends:
- Mock: CPU-only implementations (MockTrainer, MockSampler) for testing
- Tinker: Tinker SDK implementations (TinkerTrainer, TinkerSampler) for real training

All implementations use the TrainerInterface and SamplerInterface, providing
a unified API that matches the Tinker pattern:
- trainer.forward_backward(data, loss_fn, datastream_id=None)
- trainer.optim_step(optimizer_config=None)
- trainer.save_weights_and_get_sampling_client()
- sampler.sample(input_data, sampling_params, greedy=False)
"""
import os
from pathlib import Path
from typing import Optional

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    _default_grpo_save_state,
    scale_rewards,
)
from nemo_rl.algorithms.grpo_refactored import grpo_train_refactored
from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn, ClippedPGLossDataDict
from nemo_rl.algorithms.trainer_sampler import (
    TrainerInterface,
    SamplerInterface,
)
from nemo_rl.algorithms.trainer_sampler_mock import (
    MockTrainer,
    MockSampler,
    create_mock_trainer,
    create_mock_sampler,
    mock_run_multi_turn_rollout,
)
from nemo_rl.algorithms.trainer_sampler_tinker import (
    TinkerTrainer,
    TinkerSampler,
    create_tinker_trainer,
    create_tinker_sampler,
    tinker_run_multi_turn_rollout,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from transformers import AutoProcessor, PreTrainedTokenizerBase


class MockTokenizer:
    """Mock tokenizer for CPU testing."""

    def __init__(self, vocab_size: int = 1000):
        """Initialize mock tokenizer."""
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def __call__(self, text, **kwargs):
        """Tokenize text (returns random token IDs for testing)."""
        # Return random tokens
        length = len(text.split()) if isinstance(text, str) else 10
        return {"input_ids": torch.randint(0, self.vocab_size, (1, length))}

    def encode(self, text, **kwargs):
        """Encode text."""
        return self(text)["input_ids"][0].tolist()

    def decode(self, token_ids, **kwargs):
        """Decode token IDs."""
        return f"[MOCK_TOKEN_{len(token_ids)}]"


class MockLossFn(LossFunction):
    """Mock loss function for CPU testing."""

    def __call__(self, logits, data, *args, **kwargs):
        """Compute mock loss."""
        batch_size = logits.shape[0]
        # Simple cross-entropy with dummy targets
        targets = torch.randint(0, logits.shape[-1], (batch_size,))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return loss, {"mock_loss": loss.item()}


def create_mock_dataset(num_samples: int = 10, vocab_size: int = 1000):
    """Create a mock dataset for testing."""
    from nemo_rl.data.interfaces import DatumSpec

    # Create simple mock data matching DatumSpec
    samples = []
    for i in range(num_samples):
        sample: DatumSpec = {
            "message_log": [
                {
                    "role": "user",
                    "content": f"Question {i}",
                    "token_ids": torch.randint(0, vocab_size, (10,)),
                }
            ],
            "task_name": "mock_task",
            "extra_env_info": {},
            "length": 10,
            "loss_multiplier": 1.0,
            "idx": i,
        }
        samples.append(sample)

    # Create a simple dataset wrapper
    class MockDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    return MockDataset(samples)


def run_grpo_local_mock(
    master_config: Optional[MasterConfig] = None,
    num_samples: int = 10,
    max_steps: int = 2,
    max_epochs: int = 1,
    vocab_size: int = 1000,
) -> None:
    """Run GRPO training locally using mock Trainer and Sampler (CPU-only).

    This function creates mock implementations of all components and runs
    GRPO training on CPU without requiring real GPUs or model initialization.

    Args:
        master_config: Optional master configuration. If None, creates a minimal config.
        num_samples: Number of mock samples to create
        max_steps: Maximum number of training steps
        max_epochs: Maximum number of epochs
        vocab_size: Vocabulary size for mock models
    """
    print("=" * 60)
    print("  GRPO LOCAL MOCK RUNNER (CPU-ONLY)")
    print("=" * 60)
    print(f"\n▶ Configuration:")
    print(f"   - Num samples: {num_samples}")
    print(f"   - Max steps: {max_steps}")
    print(f"   - Max epochs: {max_epochs}")
    print(f"   - Vocab size: {vocab_size}")

    # Set CPU mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(4)  # Limit threads for testing

    # Initialize Ray (local mode)
    print("\n▶ Initializing Ray for local execution...")
    try:
        init_ray()
        print("  ✓ Ray initialized")
    except Exception as e:
        print(f"  ⚠️  Ray initialization warning: {e}")
        print("  Continuing with mock mode...")

    # Create mock tokenizer
    print("\n▶ Creating mock tokenizer...")
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    print("  ✓ Mock tokenizer created")

    # Create mock dataset
    print("\n▶ Creating mock dataset...")
    mock_dataset = create_mock_dataset(num_samples=num_samples, vocab_size=vocab_size)
    print(f"  ✓ Mock dataset created with {len(mock_dataset)} samples")

    # Create dataloader
    print("\n▶ Creating dataloader...")
    dataloader = StatefulDataLoader(
        mock_dataset,
        batch_size=min(4, num_samples),
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=0,  # No workers for CPU testing
    )
    print(f"  ✓ Dataloader created with batch_size={dataloader.batch_size}")

    # Create minimal master config if not provided
    if master_config is None:
        print("\n▶ Creating minimal master config...")
        master_config_dict = {
            "policy": {
                "model_name": "mock_model",
                "make_sequence_length_divisible_by": 1,
                "train_global_batch_size": 4,
                "train_micro_batch_size": 2,
                "max_total_sequence_length": 2048,
                "generation": {
                    "backend": "mock",
                    "colocated": {"enabled": True},
                },
            },
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 2,
                "max_num_steps": max_steps,
                "max_num_epochs": max_epochs,
                "max_rollout_turns": 1,
                "normalize_rewards": False,
                "use_leave_one_out_baseline": False,
                "use_dynamic_sampling": False,
                "val_period": 0,
                "val_at_start": False,
                "val_batch_size": 4,
                "max_val_samples": 10,
                "seed": 42,
                "reward_scaling": {"enabled": False},
                "reward_shaping": {"enabled": False},
            },
            "loss_fn": {
                "clip_ratio": 0.2,
                "use_importance_sampling_correction": False,
            },
            "logger": {
                "log_dir": "./mock_logs",
                "wandb_enabled": False,
            },
            "checkpointing": {
                "enabled": False,
                "checkpoint_must_save_by": None,
                "save_period": 1000,
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": 1,
            },
        }
        master_config = MasterConfig(master_config_dict)  # type: ignore
        print("  ✓ Minimal config created")

    # Create mock trainer (returns MockTrainer implementing TrainerInterface)
    print("\n▶ Creating mock trainer...")
    trainer = create_mock_trainer(
        vocab_size=vocab_size,
        colocated_inference=True,
    )
    print("  ✓ Mock trainer created (MockTrainer implementing TrainerInterface)")

    # Create mock sampler (returns MockSampler implementing SamplerInterface)
    print("\n▶ Creating mock sampler...")
    sampler = create_mock_sampler(trainer=trainer, vocab_size=vocab_size)
    print("  ✓ Mock sampler created (MockSampler implementing SamplerInterface)")

    # Create mock loss function
    print("\n▶ Creating mock loss function...")
    loss_fn = MockLossFn()
    print("  ✓ Mock loss function created")

    # Create mock logger
    print("\n▶ Creating mock logger...")
    default_logger_config = {
        "log_dir": "./mock_logs",
        "wandb_enabled": False,
        "swanlab_enabled": False,
        "tensorboard_enabled": False,
        "mlflow_enabled": False,
        "wandb": {},
        "monitor_gpus": False,
        "gpu_monitoring": {
            "collection_interval": 1.0,
            "flush_interval": 10.0,
        },
    }
    logger_config = master_config.get("logger", default_logger_config)
    # Ensure all required keys are present
    for key, value in default_logger_config.items():
        if key not in logger_config:
            logger_config[key] = value
    logger = Logger(logger_config)
    print("  ✓ Logger created")

    # Create mock checkpointer
    print("\n▶ Creating mock checkpointer...")
    default_checkpointing_config = {
        "enabled": False,
        "checkpoint_dir": "./mock_checkpoints",
        "metric_name": None,
        "higher_is_better": True,
        "save_period": 1000,
        "keep_top_k": 5,
        "checkpoint_must_save_by": None,
        "model_save_format": "safetensors",
        "max_total_sequence_length": 32_756,
        "save_consolidated": False,
        "model_cache_dir": "",
        "model_repo_id": "",
        "is_peft": False,
        "peft_config": None,
    }
    checkpointing_config = master_config.get("checkpointing", default_checkpointing_config)
    # Ensure all required keys are present
    for key, value in default_checkpointing_config.items():
        if key not in checkpointing_config:
            checkpointing_config[key] = value
    checkpointer = CheckpointManager(checkpointing_config)
    print("  ✓ Checkpointer created")

    # Create initial save state
    grpo_save_state = _default_grpo_save_state()

    # Mock task_to_env (empty for mock)
    task_to_env: dict[str, EnvironmentInterface] = {}
    val_task_to_env: dict[str, EnvironmentInterface] = {}

    print("\n▶ Starting GRPO training with mock implementations...")
    print("=" * 60)

    # Monkey-patch run_multi_turn_rollout to use mock version
    import nemo_rl.algorithms.grpo_refactored as grpo_refactored_module

    original_rollout = grpo_refactored_module.run_multi_turn_rollout
    grpo_refactored_module.run_multi_turn_rollout = mock_run_multi_turn_rollout

    try:
        # Run training using the refactored function
        grpo_train_refactored(
            trainer=trainer,
            sampler=sampler,
            dataloader=dataloader,
            val_dataloader=None,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_save_state,
            master_config=master_config,
            processor=None,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Restore original function
        grpo_refactored_module.run_multi_turn_rollout = original_rollout
        print("\n▶ Cleaning up...")
        print("  ✓ Cleanup complete")

    print("\n" + "=" * 60)
    print("  MOCK GRPO TRAINING COMPLETE")
    print("=" * 60)


def run_grpo_local_tinker(
    base_url: Optional[str] = None,
    model_name: str = "meta-llama/Llama-3.1-8B",
    lora_rank: int = 32,
    master_config: Optional[MasterConfig] = None,
    max_steps: Optional[int] = 2,
    max_epochs: Optional[int] = 1,
    vocab_size: int = 1000,
    num_samples: int = 10,
    resume_state_path: Optional[str] = None,
) -> None:
    """Run GRPO training locally using Tinker SDK implementations.

    This function sets up a minimal GRPO configuration and uses Tinker SDK
    (TrainingClient and SamplingClient) to run the refactored training loop.

    Args:
        base_url: Base URL for Tinker service (None for default)
        model_name: Model name/identifier (e.g., "meta-llama/Llama-3.1-8B")
        lora_rank: LoRA rank for training
        master_config: Optional MasterConfig to override default config.
        max_steps: Maximum number of training steps.
        max_epochs: Maximum number of training epochs.
        vocab_size: Vocabulary size for models and tokenizer.
        num_samples: Number of samples in the dataset.
        resume_state_path: Optional path to resume from checkpoint.
    """
    print("\n" + "=" * 60)
    print(" " * 18 + "STARTING TINKER GRPO LOCAL RUN")
    print("=" * 60 + "\n")

    # Initialize Ray (will use local CPU resources)
    print("▶ Initializing Ray for local execution...")
    init_ray()
    print("  ✓ Ray initialized")

    # Create minimal master config if not provided
    if master_config is None:
        print("\n▶ Creating minimal master config...")
        master_config_dict = {
            "policy": {
                "model_name": "tinker_model",
                "make_sequence_length_divisible_by": 1,
                "train_global_batch_size": 4,
                "train_micro_batch_size": 2,
                "max_total_sequence_length": 2048,
                "generation": {
                    "backend": "tinker",
                    "colocated": {"enabled": True},
                },
            },
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 2,
                "max_num_steps": max_steps,
                "max_num_epochs": max_epochs,
                "max_rollout_turns": 1,
                "normalize_rewards": False,
                "use_leave_one_out_baseline": False,
                "use_dynamic_sampling": False,
                "val_period": 0,
                "val_at_start": False,
                "val_batch_size": 4,
                "max_val_samples": 10,
                "seed": 42,
                "reward_scaling": {"enabled": False},
                "reward_shaping": {"enabled": False},
            },
            "loss_fn": {
                "clip_ratio": 0.2,
                "use_importance_sampling_correction": False,
            },
            "logger": {
                "log_dir": "./tinker_logs",
                "wandb_enabled": False,
            },
            "checkpointing": {
                "enabled": False,
                "checkpoint_must_save_by": None,
                "save_period": 1000,
            },
            "cluster": {
                "num_nodes": 1,
                "gpus_per_node": 1,
            },
        }
        master_config = MasterConfig(master_config_dict)  # type: ignore
        print("  ✓ Minimal config created")

    # Create Tinker trainer (returns TinkerTrainer implementing TrainerInterface)
    print("\n▶ Creating Tinker trainer...")
    trainer = create_tinker_trainer(
        base_url=base_url,
        model_name=model_name,
        lora_rank=lora_rank,
        vocab_size=vocab_size,
        colocated_inference=True,
        resume_state_path=resume_state_path,
    )
    print("  ✓ Tinker trainer created (TinkerTrainer implementing TrainerInterface)")

    # Create Tinker sampler (returns TinkerSampler implementing SamplerInterface)
    print("\n▶ Creating Tinker sampler...")
    sampler = create_tinker_sampler(
        base_url=base_url,
        trainer=trainer,
        vocab_size=vocab_size,
    )
    print("  ✓ Tinker sampler created (TinkerSampler implementing SamplerInterface)")

    # Create mock tokenizer (Tinker API may handle tokenization, but we need one for data prep)
    print("\n▶ Creating tokenizer...")
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    print("  ✓ Tokenizer created")

    # Create mock loss function (Tinker API handles actual loss computation)
    print("\n▶ Creating loss function...")
    loss_fn = MockLossFn()
    print("  ✓ Loss function created")

    # Create logger
    print("\n▶ Creating logger...")
    default_logger_config = {
        "log_dir": "./tinker_logs",
        "wandb_enabled": False,
        "swanlab_enabled": False,
        "tensorboard_enabled": False,
        "mlflow_enabled": False,
        "wandb": {},
        "monitor_gpus": False,
        "gpu_monitoring": {
            "collection_interval": 1.0,
            "flush_interval": 10.0,
        },
    }
    logger_config = master_config.get("logger", default_logger_config)
    # Ensure all required keys are present
    for key, value in default_logger_config.items():
        if key not in logger_config:
            logger_config[key] = value
    logger = Logger(logger_config)
    print("  ✓ Logger created")

    # Create checkpointer
    print("\n▶ Creating checkpointer...")
    default_checkpointing_config = {
        "enabled": False,
        "checkpoint_dir": "./tinker_checkpoints",
        "metric_name": None,
        "higher_is_better": True,
        "save_period": 1000,
        "keep_top_k": 5,
        "checkpoint_must_save_by": None,
        "model_save_format": "safetensors",
        "save_consolidated": False,
        "model_cache_dir": "",
        "model_repo_id": "",
        "is_peft": False,
        "peft_config": None,
    }
    checkpointing_config = master_config.get("checkpointing", default_checkpointing_config)
    # Ensure all required keys are present
    for key, value in default_checkpointing_config.items():
        if key not in checkpointing_config:
            checkpointing_config[key] = value
    checkpointer = CheckpointManager(checkpointing_config)
    print("  ✓ Checkpointer created")

    # Create initial save state
    grpo_save_state = _default_grpo_save_state()

    # Create mock dataset and dataloader
    print("\n▶ Creating dataset and dataloader...")
    dataset = create_mock_dataset(num_samples=num_samples, vocab_size=vocab_size)
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=master_config["grpo"]["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=0,
    )
    val_dataloader = StatefulDataLoader(
        dataset,
        batch_size=master_config["grpo"]["val_batch_size"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=0,
    )
    print("  ✓ Dataset and dataloader created")

    # Mock task_to_env (empty for Tinker - environment interactions may be via API)
    task_to_env: dict[str, EnvironmentInterface] = {}
    val_task_to_env: dict[str, EnvironmentInterface] = {}

    print("\n▶ Starting GRPO training with Tinker API implementations...")
    print("=" * 60)

    # Monkey-patch run_multi_turn_rollout to use Tinker version
    import nemo_rl.algorithms.grpo_refactored as grpo_refactored_module

    original_rollout = grpo_refactored_module.run_multi_turn_rollout
    grpo_refactored_module.run_multi_turn_rollout = tinker_run_multi_turn_rollout

    try:
        # Run training using the refactored function
        grpo_train_refactored(
            trainer=trainer,
            sampler=sampler,
            dataloader=dataloader,
            val_dataloader=None,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_save_state,
            master_config=master_config,
            processor=None,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Restore original function
        grpo_refactored_module.run_multi_turn_rollout = original_rollout
        print("\n▶ Cleaning up...")
        print("  ✓ Cleanup complete")

    print("\n" + "=" * 60)
    print("  TINKER GRPO TRAINING COMPLETE")
    print("=" * 60)


def main():
    """Main entry point for local testing (mock or Tinker)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GRPO locally with mock or Tinker API implementations"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["mock", "tinker"],
        help="Run mode: 'mock' for CPU-only mock, 'tinker' for Tinker API",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum training steps")
    parser.add_argument("--max-epochs", type=int, default=1, help="Maximum epochs")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    # Tinker-specific arguments
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Tinker service base URL (None for default)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name/identifier",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank for training",
    )
    parser.add_argument(
        "--resume-state-path",
        type=str,
        default=None,
        help="Path to resume from checkpoint",
    )
    args = parser.parse_args()

    if args.mode == "mock":
        run_grpo_local_mock(
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            max_epochs=args.max_epochs,
            vocab_size=args.vocab_size,
        )
    elif args.mode == "tinker":
        run_grpo_local_tinker(
            base_url=args.base_url,
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            num_samples=args.num_samples,
            max_steps=args.max_steps,
            max_epochs=args.max_epochs,
            vocab_size=args.vocab_size,
            resume_state_path=args.resume_state_path,
        )


if __name__ == "__main__":
    main()


""" example usage: 
# Mock mode:
uv run nemo_rl/algorithms/grpo_local_runner.py --num-samples 10 --max-steps 2 --max-epochs 1 --vocab-size 1000

# Tinker mode:
uv run nemo_rl/algorithms/grpo_local_runner.py --mode tinker --model-name meta-llama/Llama-3.1-8B --lora-rank 32 --num-samples 10 --max-steps 2 --max-epochs 1 --vocab-size 1000
"""