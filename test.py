import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


class RadarDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []

        # 读取数据
        for class_id in range(4):
            file_path = os.path.join(data_dir, f"class{class_id}.txt")

            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # 分割每行数据
                    line = line.strip().split()
                    timestamp = float(line[0])
                    features = [float(x) for x in line[1:]]

                    self.data.append((timestamp, features))
                    self.labels.append(class_id)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据集路径
data_dir = "path/to/txt/files"

# 创建数据集实例
dataset = RadarDataset(data_dir)

# 随机打乱数据集
indices = list(range(len(dataset)))
random.shuffle(indices)

output_dir = "path/to/save/indices.txt"
with open(output_dir, "w") as file:
    file.write("\n".join(str(index) for index in indices))

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
