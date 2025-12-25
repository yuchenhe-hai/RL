# no copy write yet will add later 
"""Mock implementations of Trainer and Sampler for CPU-only testing.

These mock implementations allow testing the GRPO interface without requiring
actual GPUs or real model initialization.
"""
import time
from typing import Any, Optional

import torch
from torch.nn import Linear

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.trainer_sampler import Sampler, Trainer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface


class MockPolicy(ColocatablePolicyInterface):
    """Mock policy implementation for CPU testing."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        """Initialize mock policy with simple linear layers."""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # Simple model for testing
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.linear = Linear(hidden_size, vocab_size)
        self.model = torch.nn.Sequential(self.embedding, self.linear)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self._weight_version = 0
        
        # Mock sharding annotations
        class MockShardingAnnotations:
            def get_axis_size(self, axis_name):
                return 1
        
        self.sharding_annotations = MockShardingAnnotations()

    def get_logprobs(self, data, **kwargs):
        """Mock logprobs computation."""
        input_ids = data["input_ids"]
        batch_size, seq_len = input_ids.shape

        # Simple forward pass
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings.mean(dim=1))  # Average pooling
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Return in expected format
        from nemo_rl.models.policy.interfaces import LogprobOutputSpec

        return BatchedDataDict(
            {
                "logprobs": logprobs.unsqueeze(1).expand(-1, seq_len, -1).mean(dim=-1),
            }
        )

    def get_reference_policy_logprobs(self, data, **kwargs):
        """Mock reference logprobs (same as current for testing)."""
        result = self.get_logprobs(data, **kwargs)
        from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec

        return BatchedDataDict({"reference_logprobs": result["logprobs"]})

    def get_topk_logits(self, data, k, **kwargs):
        """Mock topk logits."""
        input_ids = data["input_ids"]
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings.mean(dim=1))
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        from nemo_rl.models.policy.interfaces import TopkLogitsOutputSpec

        return BatchedDataDict(
            {
                "topk_logits": topk_vals,
                "topk_indices": topk_idx,
            }
        )

    def train(self, data, loss_fn, **kwargs):
        """Mock training - just do a simple forward/backward pass."""
        input_ids = data["input_ids"]
        batch_size = input_ids.shape[0]

        # Simple forward pass
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings.mean(dim=1))

        # Compute current entropy from logits (using softmax + entropy formula)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        current_entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Compute target entropy (uniform distribution over vocab)
        target_entropy = torch.log(torch.tensor(float(self.vocab_size), device=logits.device))

        # Print entropy information
        print(f"  [Mock Trainer] Current entropy: {current_entropy.item():.4f}, Target entropy: {target_entropy.item():.4f}")

        # Use a simple dummy loss that depends on model parameters but avoids cross-entropy
        # This is just a small regularization term to allow gradients to flow
        dummy_loss = logits.mean() * 0.001

        # Backward and optimizer step
        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        self._weight_version += 1

        return {
            "loss": torch.tensor([dummy_loss.item()]),
            "grad_norm": torch.tensor([1.0]),
            "all_mb_metrics": {},
        }

    def prepare_for_training(self, *args, **kwargs):
        """Mock prepare for training."""
        self.model.train()
        print("  [MOCK] Policy prepared for training")

    def prepare_for_lp_inference(self, *args, **kwargs):
        """Mock prepare for logprob inference."""
        self.model.eval()
        print("  [MOCK] Policy prepared for logprob inference")

    def finish_training(self, *args, **kwargs):
        """Mock finish training."""
        pass

    def save_checkpoint(self, *args, **kwargs):
        """Mock checkpoint saving."""
        print("  [MOCK] Checkpoint saved")

    def shutdown(self):
        """Mock shutdown."""
        return True

    def init_collective(self, *args, **kwargs):
        """Mock collective init."""
        return []

    def offload_before_refit(self, *args, **kwargs):
        """Mock offload before refit."""
        pass

    def offload_after_refit(self, *args, **kwargs):
        """Mock offload after refit."""
        pass

    def prepare_refit_info(self, *args, **kwargs):
        """Mock prepare refit info."""
        return {"weight_version": self._weight_version}

    def stream_weights_via_ipc_zmq(self, *args, **kwargs):
        """Mock weight streaming."""
        print("  [MOCK] Weight streaming via IPC ZMQ")
        return []

    def broadcast_weights_for_collective(self, *args, **kwargs):
        """Mock weight broadcast."""
        print("  [MOCK] Weight broadcast via collective")
        return []

    def calibrate_qkv_fp8_scales(self, *args, **kwargs):
        """Mock KV scale calibration."""
        return {"layers": {}}

    def get_free_memory_bytes(self):
        """Mock free memory."""
        return 8 * 1024**3  # 8GB

    def print_node_ip_and_gpu_id(self):
        """Mock print node info."""
        print("  [MOCK] Node: localhost, GPU: N/A (CPU mode)")


class MockGeneration(GenerationInterface):
    """Mock generation implementation for CPU testing."""

    def __init__(self, vocab_size: int = 1000):
        """Initialize mock generation."""
        self.vocab_size = vocab_size
        self._weight_version = 0
        self._stale = True

    def init_collective(self, *args, **kwargs):
        """Mock collective init."""
        return []

    def generate(self, data, greedy=False):
        """Mock generation - return random token sequences."""
        input_ids = data["input_ids"]
        batch_size, input_len = input_ids.shape
        max_new_tokens = 10

        # Generate random output tokens
        output_ids = torch.randint(0, self.vocab_size, (batch_size, input_len + max_new_tokens))
        output_ids[:, :input_len] = input_ids  # Copy input

        generation_lengths = torch.full((batch_size,), max_new_tokens, dtype=torch.long)
        unpadded_lengths = input_len + generation_lengths

        # Mock logprobs
        logprobs = torch.randn(batch_size, input_len + max_new_tokens) * 0.1

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids,
                "generation_lengths": generation_lengths,
                "unpadded_sequence_lengths": unpadded_lengths,
                "logprobs": logprobs,
            }
        )

    def prepare_for_generation(self, *args, **kwargs):
        """Mock prepare for generation."""
        print("  [MOCK] Generation prepared")
        self._stale = False

    def finish_generation(self, *args, **kwargs):
        """Mock finish generation."""
        pass

    def prepare_refit_info(self, state_dict_info=None):
        """Mock prepare refit info."""
        if state_dict_info:
            self._weight_version = state_dict_info.get("weight_version", 0)

    def update_weights_via_ipc_zmq(self):
        """Mock weight update via IPC."""
        print("  [MOCK] Weights updated via IPC ZMQ")
        self._stale = False
        return []

    def update_weights_from_collective(self):
        """Mock weight update from collective."""
        print("  [MOCK] Weights updated from collective")
        self._stale = False
        return []

    def requires_kv_scale_sync(self):
        """Mock KV scale sync requirement (property)."""
        return False


def create_mock_trainer(
    vocab_size: int = 1000,
    hidden_size: int = 128,
    colocated_inference: bool = False,
) -> Trainer:
    """Create a mock trainer for CPU testing.

    Args:
        vocab_size: Vocabulary size for mock model
        hidden_size: Hidden size for mock model
        colocated_inference: Whether inference is colocated

    Returns:
        Trainer instance with mock policy
    """
    from nemo_rl.algorithms.trainer_sampler import Trainer, TrainerConfig

    policy = MockPolicy(vocab_size=vocab_size, hidden_size=hidden_size)

    # Create a mock refit function
    def mock_refit_fn(
        policy_interface,
        generation_interface,
        colocated,
        buffer_size_gb=None,
        timer=None,
        kv_scales=None,
    ):
        """Mock refit function that simulates weight sync."""
        print(f"  [MOCK] Refitting generation with policy weights (colocated={colocated})")
        if timer:
            with timer.time("mock_weight_sync"):
                time.sleep(0.01)  # Simulate weight transfer time
        else:
            time.sleep(0.01)

        # Update generation weights
        if hasattr(generation_interface, "prepare_refit_info"):
            policy_refit_info = policy_interface.prepare_refit_info()
            generation_interface.prepare_refit_info(policy_refit_info)

        # Simulate weight update
        if colocated:
            generation_interface.update_weights_via_ipc_zmq()
        else:
            generation_interface.update_weights_from_collective()

        print("  [MOCK] Weight sync complete")

    trainer = Trainer(
        policy=policy,
        colocated_inference=colocated_inference,
        refit_fn=mock_refit_fn,
    )

    return trainer


def create_mock_sampler(
    trainer: Optional[Trainer] = None,
    vocab_size: int = 1000,
) -> Sampler:
    """Create a mock sampler for CPU testing.

    Args:
        trainer: Optional trainer to link for weight sync
        vocab_size: Vocabulary size for mock generation

    Returns:
        Sampler instance with mock generation
    """
    from nemo_rl.algorithms.trainer_sampler import Sampler, SamplingConfig

    generation = MockGeneration(vocab_size=vocab_size)

    sampler = Sampler(generation=generation, trainer=trainer)

    return sampler


def mock_run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch,
    tokenizer,
    task_to_env,
    max_seq_len: int,
    max_rollout_turns: int = 1,
    greedy: bool = False,
):
    """Mock version of run_multi_turn_rollout that doesn't require real environments.

    This simulates the rollout by:
    1. Generating responses using the generation interface
    2. Adding mock rewards to the batch
    3. Returning the batch with rewards and mock metrics

    Args:
        policy_generation: Generation interface
        input_batch: Input batch with message logs
        tokenizer: Tokenizer (not used in mock)
        task_to_env: Task to environment mapping (not used in mock)
        max_seq_len: Max sequence length
        max_rollout_turns: Max rollout turns (mock uses 1)
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (updated_batch, rollout_metrics)
    """
    import copy
    from nemo_rl.data.interfaces import DatumSpec
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import GenerationDatumSpec

    # Copy batch to avoid modifying original
    batch = copy.deepcopy(input_batch)

    # Convert message log to flat format for generation
    batched_flat, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0},
    )
    input_ids = batched_flat["token_ids"]

    # Prepare generation input
    generation_input = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )
    generation_input.to("cpu")

    # Generate responses
    output = policy_generation.generate(generation_input, greedy=greedy)

    # Update message log with generated responses (simplified - just add assistant message)
    batch_size = len(batch["message_log"])
    for i in range(batch_size):
        # Extract generated tokens for this sample
        gen_length = output["generation_lengths"][i].item()
        gen_start = input_lengths[i].item()
        gen_end = gen_start + gen_length
        gen_tokens = output["output_ids"][i, gen_start:gen_end].cpu()

        # Add assistant message with generated tokens
        assistant_msg = {
            "role": "assistant",
            "token_ids": gen_tokens,
            "generation_logprobs": output["logprobs"][i, gen_start:gen_end].cpu(),
        }
        batch["message_log"][i].append(assistant_msg)

    # Add mock rewards (random rewards for testing)
    batch["total_reward"] = torch.rand(batch_size, dtype=torch.float32) * 2.0 - 1.0  # -1 to 1
    batch["loss_multiplier"] = torch.ones(batch_size, dtype=torch.float32)
    batch["length"] = torch.tensor([input_lengths[i].item() for i in range(batch_size)], dtype=torch.long)

    # Mock rollout metrics
    rollout_metrics = {
        "mean_gen_tokens_per_sample": float(output["generation_lengths"].float().mean().item()),
        "total_gen_tokens": int(output["generation_lengths"].sum().item()),
    }

    return batch, rollout_metrics

