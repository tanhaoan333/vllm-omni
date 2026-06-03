# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
_MODEL_PATCHES_APPLIED = False


def apply_npu_model_patches() -> None:
    """Register NPU-specific model monkey-patches (idempotent)."""
    global _MODEL_PATCHES_APPLIED
    if _MODEL_PATCHES_APPLIED:
        return
    from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import apply_qwen3_tts_code2wav_patch

    apply_qwen3_tts_code2wav_patch()
    _MODEL_PATCHES_APPLIED = True
