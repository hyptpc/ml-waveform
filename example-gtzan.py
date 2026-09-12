#!/usr/bin/env python3

'''
Example code of CNN model training with GTZAN sample data.
'''

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import GTZAN
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import Resize

import random_seed

top_dir = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(top_dir, 'data')
# GTZAN is distributed at 22.05 kHz; resample to a lighter rate.
ORIG_FREQ = 22050
NEW_FREQ = 16000
# n_fft must give at least n_mels frequency bins, otherwise some mel
# filterbanks come out all zero.
N_FFT = 1024
HOP_LENGTH = 160
# Every clip is squeezed into this fixed spectrogram shape so that the
# fully connected layer has a known input size.
N_MELS = 128
N_FRAMES = 250
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
POOL_KERNEL = 2
N_POOL_LAYERS = 2
GENRES = ('blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock')
N_CLASSES = len(GENRES)
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

#______________________________________________________________________________
class SpectrogramDataset(Dataset):
  ''' turn GTZAN items into (spectrogram, class index) pairs '''

  def __init__(self, dataset, indices):
    self.dataset = dataset
    self.indices = indices
    self.resample = Resample(orig_freq=ORIG_FREQ, new_freq=NEW_FREQ)
    self.mel_spectrogram = MelSpectrogram(n_fft=N_FFT,
                                          hop_length=HOP_LENGTH,
                                          n_mels=N_MELS)
    self.resize = Resize((N_MELS, N_FRAMES), antialias=True)

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, index):
    waveform, _, label = self.dataset[self.indices[index]]
    spectrogram = self.mel_spectrogram(self.resample(waveform))
    return self.resize(spectrogram), GENRES.index(label)

#______________________________________________________________________________
class CNN(nn.Module):
  ''' two convolution blocks followed by a linear classifier '''

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, CONV1_CHANNELS, kernel_size=(3, 3),
                           stride=(1, 1), padding=(1, 1))
    self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS,
                           kernel_size=(3, 3), stride=(1, 1),
                           padding=(1, 1))
    self.pool = nn.MaxPool2d(kernel_size=(POOL_KERNEL, POOL_KERNEL),
                             stride=(POOL_KERNEL, POOL_KERNEL))
    divisor = POOL_KERNEL**N_POOL_LAYERS
    n_features = (CONV2_CHANNELS * (N_MELS // divisor)
                  * (N_FRAMES // divisor))
    self.fc1 = nn.Linear(n_features, N_CLASSES)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.flatten(1) # keep the batch dimension
    return self.fc1(x)

#______________________________________________________________________________
def genre_labels(dataset):
  ''' genre of every item, read from the GTZAN file names

  Going through __getitem__ would decode all 1000 audio files just to
  collect the labels, so use the file names instead.
  '''
  walker = getattr(dataset, '_walker', None)
  if walker is None:
    return None
  return [os.path.basename(str(f)).split('.')[0] for f in walker]

#______________________________________________________________________________
def build_loaders(root):
  ''' split GTZAN into train/test loaders '''
  os.makedirs(root, exist_ok=True)
  dataset = GTZAN(root=root, download=True)
  train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=genre_labels(dataset))
  generator = random_seed.make_generator(RANDOM_STATE)
  train_loader = DataLoader(
    SpectrogramDataset(dataset, train_indices), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS,
    worker_init_fn=random_seed.seed_worker, generator=generator)
  test_loader = DataLoader(
    SpectrogramDataset(dataset, test_indices), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS,
    worker_init_fn=random_seed.seed_worker)
  return train_loader, test_loader

#______________________________________________________________________________
def evaluate(model, loader, device):
  ''' accuracy over the given loader '''
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for data, labels in loader:
      data, labels = data.to(device), labels.to(device)
      _, predicted = torch.max(model(data), 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return correct / total if total > 0 else 0.

#______________________________________________________________________________
def run(root=DATA_DIR, num_epochs=NUM_EPOCHS):
  ''' train the CNN and report the test accuracy of each epoch '''
  random_seed.set_seed(RANDOM_STATE)
  train_loader, test_loader = build_loaders(root)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = CNN().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  print('Training start')
  for epoch in range(num_epochs):
    model.train()
    loss = None
    for data, labels in train_loader:
      data, labels = data.to(device), labels.to(device)
      optimizer.zero_grad()
      loss = criterion(model(data), labels)
      loss.backward()
      optimizer.step()
    accuracy = evaluate(model, test_loader, device)
    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Loss: {loss.item() if loss is not None else float("nan")}, '
          f'Accuracy: {accuracy}')
  print('Training finished')

#______________________________________________________________________________
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', default=DATA_DIR,
                      help='directory where GTZAN is downloaded')
  parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                      help='number of training epochs')
  parsed, unparsed = parser.parse_known_args()
  run(root=parsed.root, num_epochs=parsed.epochs)
