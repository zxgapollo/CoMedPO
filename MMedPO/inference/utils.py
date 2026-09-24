import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch
import os
import json
from PIL import Image
import sys
import random
import numpy as np

sys.path.append("../train/dpo/llava")



class QuestionDataset(Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    # def __getitem__(self, idx):
    #     return self.questions[idx]
    def __getitem__(self, idx):
    # 获取原始的数据记录
        record = self.questions[idx]
        
        # 检查 'image_path' 字段
        image_path_value = record.get("image_path", record.get("image"))
        
        # 如果它是一个列表，只取第一个元素
        if isinstance(image_path_value, list) and image_path_value:
            record["image_path"] = image_path_value[0]
        # 如果它是其他类型但存在，保持原样（以防万一有单一张图片的数据）
        elif image_path_value:
            record["image_path"] = image_path_value
        
        # 返回修改后的记录
        return record


class QuestionDataset_fromGTblank(Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]



def setup():
    # 检查是否在分布式环境中
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # 单GPU模式，设置环境变量
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        # 不初始化分布式进程组



def cleanup():
    # 只在分布式模式下销毁进程组
    if dist.is_initialized():
        dist.destroy_process_group()


def tensor_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    return obj
