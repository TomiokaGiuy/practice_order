import torch
import torch.nn as nn
import torchvision
from pandas.core.dtypes.common import classes
from torchvision import datasets, transforms

from torchvision.datasets import  ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.transforms import v2

import os
from tqdm import  tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import struct
import sys

from array import array
from os import path



def rename_images_in_folder(folder_path):
    files = os.listdir(folder_path)

    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    image_files.sort()

    for index, filename in enumerate(image_files, start=1):
        file_extension = os.path.splitext(filename)[1]

        new_name = f"{index}{file_extension}"

        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)

        os.rename(old_file, new_file)
        print(f"Переименован: {filename} -> {new_name}")

    print("Все файлы переименованы!")



class CattleDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        self.data_list = []
        self.len_data = 0

        for path_dir, dir_list, file_list in os.walk(path):
            # print("path_dir " + path_dir )
            normalized_path = os.path.normpath(path_dir)
            cls = normalized_path.split(os.sep)[-1]
            # print("class " + cls)
            # print("class _  " + path_dir)

            for file_name in file_list:
                file_path = os.path.join(path_dir, file_name)
                # print("file_path " + file_path)
                self.data_list.append((file_path, cls))


            self.len_data +=len(file_list)
            print("len_data " + str(self.len_data))

    def __len__(self):
        return self.len_data

    def __getitem__(self, item):
        file_path, target = self.data_list[item]

        class_to_idx = {"healthy": 0, "lumpy": 1}
        target = class_to_idx[target]

        sample = Image.open(file_path).convert("RGB")  # Гарантия 3-канального изображения
        sample = np.array(sample)

        if self.transform is not None:
            sample = self.transform(sample)


        return sample, target

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device " + device)
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((128, 128)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

data_healthy = CattleDataset("dataset/main/healthy", transform)
data_lumpy = CattleDataset("dataset/main/lumpy", transform)


train_healthy, val_healthy, test_healthy = random_split(data_healthy, [0.8, 0.2, 0])
train_lumpy, val_lumpy, test_lumpy = random_split(data_lumpy, [0.8, 0.2, 0])


train_data = ConcatDataset([train_healthy, train_lumpy])
val_data = ConcatDataset([val_healthy, val_lumpy])
test_data = ConcatDataset([test_healthy, test_lumpy])

# протестить с разным кол-вом батчей 32, 64... больше/меньше
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Сделать архитектуру сверточной нейроной сети
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Новый Dropout снижает переобучение
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.model(x)
        return out

model = MyModel().to(device)


input = torch.rand([16, 3, 128, 128], dtype=torch.float32).to(device)

out = model(input)
print(out.shape) #16 10

loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 25
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for epoch in range(EPOCHS):
    #Train model
    model.train() # always before train
    running_train_loss = []
    true_answer = 0
    train_loop = tqdm(train_loader, leave=False) #понимать на какой стадии обучение, False нужно что бы прогрессбар удалялся после каждой итерации
    for x, targets in train_loop: # при добавление train_loop после in  мы им заменяем train_loader
        #   Данные
        #   (batch_size, 1, 28, 28) -> (batch_size, 784)
        # x = x.reshape(-1, 28*28).to(device).to(torch.float32)
        x = x.to(device).to(torch.float32)  # Оставляем в формате (batch, 3, 128, 128)

        #   (batch_size, int) -> (batch_size, 10), dtype=float32
        # targets = targets.reshape(-1).to(torch.int32)
        targets = targets.reshape(-1).to(torch.long)  # Используем long для меток

        # targets = torch.eye(10)[targets].to(device) #targets = targets.to(device)
        targets = targets.to(device)
        #targets = targets.to(device)

        #   Прямой проход + расчет ошибки модели
        pred = model(x)
        loss = loss_model(pred, targets)

        #   Обратный проход
        opt.zero_grad()
        loss.backward()
        #   Шаг оптимизации
        opt.step()

        #для вывода на экран какая эпоха и какие потери
        running_train_loss.append(loss.item())
        mean_train_loss= sum(running_train_loss)/len(running_train_loss)

        # true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
        true_answer += (pred.argmax(dim=1) == targets).sum().item()

        train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}, train_loss={mean_train_loss:.4f}")



    #   Расчет значения метрики
    running_train_acc = true_answer/ len(train_data)

    #   Сохранение значения функции потерь и метрики
    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)



    # Проверка модели (валидация)
    model.eval() # всегда перед валидации
    with torch.no_grad(): #запрещаем использование градиентов ОБЯЗАТЕЛЬНО
        running_val_loss = []
        true_answer = 0
        for x, targets in val_loader:
            #   Данные
            #   (batch_size, 1, 28, 28) -> (batch_size, 784)
            # x = x.reshape(-1, 28 * 28).to(device).to(torch.float32)
            x = x.to(device).to(torch.float32)
            #   (batch_size, int) -> (batch_size, 10), dtype=float32
            targets = targets.to(torch.long).to(device)

            # targets = torch.eye(10)[targets].to(device)

            #   Прямой проход + расчет ошибки модели
            pred = model(x).to(torch.float32)
            loss = loss_model(pred, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

            # true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            true_answer += (pred.argmax(dim=1) == targets).sum().item()

        #   Расчет значений метрики
        running_val_acc = true_answer / len(val_data)
        #   Сохранение значения функции потерь и метрики
        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}, train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}")

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(["loss_train", "loss_val"])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['acc_train', 'acc_val'])
plt.show()


