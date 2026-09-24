#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            # 动态添加所有缺失的LlamaConfig属性以兼容新版本transformers
            from transformers.models.llama.configuration_llama import LlamaConfig
            llama_config = LlamaConfig.from_pretrained(model_path)
            # 将缺失的属性从LlamaConfig复制到当前配置
            for attr in dir(llama_config):
                if not attr.startswith('_') and not hasattr(lora_cfg_pretrained, attr):
                    setattr(lora_cfg_pretrained, attr, getattr(llama_config, attr))
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            print('Loading non-LoRA trainables...')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                model.load_state_dict(non_lora_state_dict, strict=False)
            print('Model is loaded...')
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                # 添加所有缺失的属性以兼容新版本transformers
                for attr in ['attention_dropout', 'rope_theta', 'rms_norm_eps', 'use_cache', 'tie_word_embeddings']:
                    if not hasattr(cfg_pretrained, attr):
                        if attr == 'attention_dropout':
                            cfg_pretrained.attention_dropout = 0.0
                        elif attr == 'rope_theta':
                            cfg_pretrained.rope_theta = 10000.0
                        elif attr == 'rms_norm_eps':
                            cfg_pretrained.rms_norm_eps = 1e-6
                        elif attr == 'use_cache':
                            cfg_pretrained.use_cache = True
                        elif attr == 'tie_word_embeddings':
                            cfg_pretrained.tie_word_embeddings = False
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                # 如果model_base存在，从model_base加载tokenizer，否则从model_path加载
                if model_base is not None:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                
                if model_base is not None:
                    # 先加载基础模型，再把 model_path 作为 LoRA 适配器插上去
                    print('Loading base model for LLaVA-Mistral...')
                    model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
                    print(f'Loading non-LoRA trainables from {model_path}')
                    non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
                    if os.path.exists(non_lora_path):
                        non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                        model.load_state_dict(non_lora_state_dict, strict=False)
                    print('Model is loaded...')
                else:
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    # 动态添加所有缺失的LlamaConfig属性以兼容新版本transformers
                    from transformers.models.llama.configuration_llama import LlamaConfig
                    llama_config = LlamaConfig.from_pretrained(model_path)
                    # 将缺失的属性从LlamaConfig复制到当前配置
                    for attr in dir(llama_config):
                        if not attr.startswith('_') and not hasattr(cfg_pretrained, attr):
                            setattr(cfg_pretrained, attr, getattr(llama_config, attr))
                    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # Try to load mm_projector.bin first, if not found, try non_lora_trainables.bin
            mm_projector_path = os.path.join(model_path, 'mm_projector.bin')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            
            if os.path.exists(mm_projector_path):
                print(f'Loading mm_projector weights from {mm_projector_path}')
                mm_projector_weights = torch.load(mm_projector_path, map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            elif os.path.exists(non_lora_path):
                print(f'Loading non-LoRA trainables from {non_lora_path}')
                non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                model.load_state_dict(non_lora_state_dict, strict=False)
            else:
                print(f'Warning: No mm_projector.bin or non_lora_trainables.bin found in {model_path}')
                print('Proceeding without additional weights...')
        else:
            if 'mpt' in model_name.lower():
                # 如果model_base存在，从model_base加载tokenizer，否则从model_path加载
                if model_base is not None:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                # 如果model_base存在，从model_base加载tokenizer，否则从model_path加载
                if model_base is not None:
                    tokenizer = AutoTokenizer.from_pretrained(model_base)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                # 如果model_base存在，从model_base加载tokenizer，否则从model_path加载
                if model_base is not None:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # 根据模型类型选择正确的类
            if 'mpt' in model_name.lower():
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading non-LoRA trainables from {model_path}")
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                model.load_state_dict(non_lora_state_dict, strict=False)
            print('Model is loaded...')
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
