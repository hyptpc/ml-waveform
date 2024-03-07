#!/usr/bin/env python3

'''
Example code of RNN model training
'''

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

#______________________________________________________________________________
def generate_sample_data(num_samples=1000):
  ''' generate sample data '''
  # random data
  amplitudes = torch.rand(num_samples) * 10.0  # 0-10
  times = torch.rand(num_samples) * 5.0  # 0-5
  # sin sampling
  sample_rate = 1000
  t = torch.arange(0, 5, 1/sample_rate)
  signals = [a * torch.sin(2 * torch.pi * t / T) for a, T in zip(amplitudes, times)]
  # add noise
  noise = torch.randn_like(t)  # normal dist
  signals_with_noise = [signal + 0.1 * noise for signal in signals]
  return signals_with_noise, torch.stack([amplitudes, times], dim=1)

#______________________________________________________________________________
class AmplitudeTimeDataset(Dataset):
  ''' data set for tuple of amplitude and time '''
  def __init__(self, signals, labels):
    self.signals = signals
    self.labels = labels

  def __len__(self):
    return len(self.signals)

  def __getitem__(self, idx):
    return self.signals[idx], self.labels[idx]

#______________________________________________________________________________
class AmplitudeTimeRNN(nn.Module):
  ''' RNN model '''
  def __init__(self):
    super(AmplitudeTimeRNN, self).__init__()
    self.rnn = nn.RNN(input_size=5000, hidden_size=64, batch_first=True)
    self.fc = nn.Linear(64, 2)  # 2 outputs, amplitude/time

  def forward(self, x):
    _, hn = self.rnn(x)
    output = self.fc(hn.squeeze(0))
    return output

#______________________________________________________________________________
if __name__ == '__main__':
  # preprocess
  signals, labels = generate_sample_data()
  dataset = AmplitudeTimeDataset(signals, labels)
  train_dataset, test_dataset = train_test_split(
    dataset, test_size=0.2, random_state=42)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  # setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = AmplitudeTimeRNN().to(device)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  step_size = 10
  gamma = 0.9
  scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  # start training
  train_loss_list = []
  test_loss_list = []
  num_epochs = 100
  for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for signals, labels in train_loader:
      signals, labels = signals.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(signals)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    train_loss_list.append(train_loss)
    scheduler.step()

    model.eval()
    with torch.no_grad():
      test_loss = 0
      for signals, labels in test_loader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        test_loss += criterion(outputs, labels).item()
    test_loss = test_loss / len(test_loader)
    test_loss_list.append(test_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {test_loss}")
  print("Training finished!")
  plt.plot(range(len(train_loss_list)), train_loss_list,
           c='b', label='train loss')
  plt.plot(range(len(test_loss_list)), test_loss_list,
           c='r', label='test loss')
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.legend()
  plt.grid()
  plt.show()
