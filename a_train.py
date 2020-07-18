# 大分类
import os
import glob
import sys
import copy
import time
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection  import train_test_split
from tqdm import tqdm
from scipy.io import wavfile

import common as com
from model import Model_a
from config import Config


def file_list_generator(target_dir, dir_name="train", ext="wav"):

    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath(
        "{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files


def load_dir(target_dir, max_length=160000):
    files = file_list_generator(target_dir)

    x = []
    for fname in tqdm(files):
        _, d = wavfile.read(fname)
        if d.shape[0] > max_length:
            d = d[0:max_length]
        x.append(d.T)

    data = np.array(x)
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=1)
    data = np.expand_dims(data, axis=-1)

    return data


def load_data(dirs):
    label = []
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        d = load_dir(target_dir)

        label.extend([idx] * d.shape[0])
        if idx == 0 :
            data = d
        else:
            data = np.vstack((data, d))

    print("data shape :", data.shape)
    label = (np.array(label)).astype(np.int64)

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1)

    train_dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label)),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label)),
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_dataloader, test_dataloader


def model_train(model, train_dataloader, test_dataloader):
    device = torch.device(Config.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=Config.step_size, gamma=Config.gamma)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    train_acc = []
    num_epochs = Config.num_epochs

    since = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        com.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)
        running_loss = 0
        running_corrects = 0

        model.train()
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)

            running_loss += loss.data.item() * inputs.size(0)
            preds = torch.argmax(out, 1)
            running_corrects += (preds == labels.data).sum().float()

            loss.backward()
            optimizer.step()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_acc = running_corrects.double() / len(train_dataloader.dataset)
        com.logger.info('train Loss: {:.4f}'.format(train_loss))
        com.logger.info('train acc :{:.4f}'.format(train_acc))

        running_loss = 0
        running_corrects = 0

        model.eval()
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            loss = criterion(out, labels)

            running_loss += loss.data.item() * inputs.size(0)
            preds = torch.argmax(out, 1)
            running_corrects += (preds == labels.data).sum().float()

        val_loss = running_loss / len(test_dataloader.dataset)
        val_acc = running_corrects.double() / len(test_dataloader.dataset)
        com.logger.info('val Loss: {:.4f}'.format(val_loss))
        com.logger.info('val acc : {:.4f}'.format(val_acc))

        if Config.stepLR: scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % 5 == 0:
            model_file_path = "{}/tmp_model{}.pkl".format(Config.model_directory, epoch)
            torch.save(model, model_file_path)
            print("temporarily save model")

        print("epoch_time: {}".format(time.time() - epoch_start))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, train_loss_history, val_loss_history


def main():
    os.makedirs(Config.model_directory, exist_ok=True)

    dirs = com.choose_dirs(Config.dev_directory)

    train_dataloader, test_dataloader = load_data(dirs)

    com.logger.info("============== MODEL TRAINING ==============")
    model = Model_a(len(dirs))
    print(model)

    model, _, _ = model_train(model, train_dataloader, test_dataloader)

    model_file_path = "{}/model_a.pkl".format(Config.model_directory)
    torch.save(model, model_file_path)
    print("save_model -> {}".format(model_file_path))
    com.logger.info("============== END TRAINING ==============")


if __name__ == "__main__":
    main()