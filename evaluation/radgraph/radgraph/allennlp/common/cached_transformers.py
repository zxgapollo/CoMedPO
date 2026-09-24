import logging
from typing import NamedTuple, Optional, Dict, Tuple
import transformers
from transformers import AutoModel, AutoConfig


logger = logging.getLogger(__name__)


class TransformerSpec(NamedTuple):
    model_name: str
    override_weights_file: Optional[str] = None
    override_weights_strip_prefix: Optional[str] = None


_model_cache: Dict[TransformerSpec, transformers.PreTrainedModel] = {}


def get(
    model_name: str,
    make_copy: bool,
    override_weights_file: Optional[str] = None,
    override_weights_strip_prefix: Optional[str] = None,
) -> transformers.PreTrainedModel:
    """
    Returns a transformer model from the cache.

    # Parameters

    model_name : `str`
        The name of the transformer, for example `"bert-base-cased"`
    make_copy : `bool`
        If this is `True`, return a copy of the model instead of the cached model itself. If you want to modify the
        parameters of the model, set this to `True`. If you want only part of the model, set this to `False`, but
        make sure to `copy.deepcopy()` the bits you are keeping.
    override_weights_file : `str`, optional
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix : `str`, optional
        If set, strip the given prefix from the state dict when loading it.
    """
    global _model_cache
    spec = TransformerSpec(model_name, override_weights_file, override_weights_strip_prefix)
    transformer = _model_cache.get(spec, None)
    if transformer is None:
        if override_weights_file is not None:
            from radgraph.allennlp.common.file_utils import cached_path
            import torch

            override_weights_file = cached_path(override_weights_file)
            override_weights = torch.load(override_weights_file)
            if override_weights_strip_prefix is not None:

                def strip_prefix(s):
                    if s.startswith(override_weights_strip_prefix):
                        return s[len(override_weights_strip_prefix) :]
                    else:
                        return s

                valid_keys = {
                    k
                    for k in override_weights.keys()
                    if k.startswith(override_weights_strip_prefix)
                }
                if len(valid_keys) > 0:
                    logger.info(
                        "Loading %d tensors from %s", len(valid_keys), override_weights_file
                    )
                else:
                    raise ValueError(
                        f"Specified prefix of '{override_weights_strip_prefix}' means no tensors "
                        f"will be loaded from {override_weights_file}."
                    )
                override_weights = {strip_prefix(k): override_weights[k] for k in valid_keys}

            transformer = AutoModel.from_pretrained(model_name, state_dict=override_weights)
        else:
            # Allow using a local HuggingFace model directory when running offline.
            # If LOCAL_HF_MODELS_DIR is set, and model_name is like "org/model",
            # try loading from LOCAL_HF_MODELS_DIR/model instead of attempting network access.
            import os
            local_root = os.environ.get("LOCAL_HF_MODELS_DIR", "/share_docker/workspace/Med/CheXbert")
            local_path = None
            if "/" in model_name:
                _, short = model_name.split("/", 1)
                candidate = os.path.join(local_root, short)
                if os.path.isdir(candidate):
                    local_path = candidate
            if local_path is not None:
                config = AutoConfig.from_pretrained(local_path)
                transformer = AutoModel.from_config(config)
                # also set model repo path for potential further loads
            else:
                config = AutoConfig.from_pretrained(model_name)
                transformer = AutoModel.from_config(config)
        _model_cache[spec] = transformer
    if make_copy:
        import copy

        return copy.deepcopy(transformer)
    else:
        return transformer


_tokenizer_cache: Dict[Tuple[str, frozenset], transformers.PreTrainedTokenizer] = {}


def get_tokenizer(model_name: str, **kwargs) -> transformers.PreTrainedTokenizer:
    cache_key = (model_name, frozenset(kwargs.items()))
    global _tokenizer_cache
    tokenizer = _tokenizer_cache.get(cache_key, None)
    if tokenizer is None:
        # Try local model directory first if available (useful when offline).
        import os
        local_root = os.environ.get("LOCAL_HF_MODELS_DIR", "/share_docker/workspace/Med/CheXbert")
        local_path = None
        if "/" in model_name:
            _, short = model_name.split("/", 1)
            candidate = os.path.join(local_root, short)
            if os.path.isdir(candidate):
                local_path = candidate
        if local_path is not None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(local_path, **kwargs)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
        # Ensure compatibility helpers for tokenizer API across transformers versions\n+        def _ensure_tokenizer_compat(tok):\n+            # encode_plus -> __call__ fallback\n+            if not hasattr(tok, \"encode_plus\") and hasattr(tok, \"__call__\"):\n+                try:\n+                    tok.encode_plus = tok.__call__\n+                except Exception:\n+                    pass\n+            # build_inputs_with_special_tokens fallback\n+            if not hasattr(tok, \"build_inputs_with_special_tokens\"):\n+                def _build_inputs_with_special_tokens(ids):\n+                    # Prefer prepare_for_model when available\n+                    if hasattr(tok, \"prepare_for_model\"):\n+                        out = tok.prepare_for_model(ids, add_special_tokens=True)\n+                        return out.get(\"input_ids\", ids)\n+                    # Last resort: return ids unchanged\n+                    return ids\n+\n+                tok.build_inputs_with_special_tokens = _build_inputs_with_special_tokens\n+\n+        _ensure_tokenizer_compat(tokenizer)\n+        _tokenizer_cache[cache_key] = tokenizer
    return tokenizer
