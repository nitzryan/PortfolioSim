import pandas as pd
from scipy.stats import zscore
import numpy as np

# Read data, seperating into what is full and what needs to be determined
df = pd.read_csv('data//financialData.csv')
no_nan_df = df[~df.isnull().any(axis=1)].copy()
no_nan_df.loc[:, '10YrChange'] = df['10YrRate'] - df['10YrRate'].shift(1)
no_nan_df = no_nan_df.drop('Year', axis=1)
no_nan_df_norm = no_nan_df.apply(zscore)
means = no_nan_df.mean()
stds = no_nan_df.std()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

correlationPlot = False
if correlationPlot:
  df = no_nan_df_norm
  varList = ["USGDP%", "Inflation$", "10YrChange", "10YrRate", "3Month", "10Yr", "Baa Corporate", "SP500", "Real Estate", "REIT", "Gold", "PE", "CAPE"]
  df = df[varList]

  f, ax = plt.subplots(figsize=(20, 20))
  df_shifted = df.shift(1)
  df_shifted = df_shifted.rename(columns=lambda x: x + ' Prev')
  df_new = pd.concat([df, df_shifted], axis=1)
  corr = df_new.corr()
  cmap = LinearSegmentedColormap.from_list('custom_cmap',[(1,0,0), (0,0,0), (0,1,0)])
  sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, annot=True,
              square=True, linewidths=.5, cbar_kws={"shrink": .5}, fmt=".2f")

  plt.show()
  exit(1)

# Get PCA components (useful for adding noise that doesn't completely change relationships)
from sklearn.decomposition import PCA
pca = PCA(n_components=len(means))
pca.fit_transform(no_nan_df_norm)
pca_vectors_df = pd.DataFrame(pca.components_, columns=no_nan_df_norm.columns)
pca_vectors_df = pca_vectors_df.assign(ExplainedStd=np.sqrt(pca.explained_variance_ratio_))
pca_vectors_df["ExplainedStd"] = pca_vectors_df["ExplainedStd"] / pca_vectors_df["ExplainedStd"].sum()

# Drop Components that have low correlation with REITs to help prevent overfitting
drop_columns = ["USGDP%", "Inflation$", "10YrChange", "10YrRate", "3Month", "10Yr", "Baa Corporate", "Gold", "PE", "CAPE"]

for column in drop_columns:
  no_nan_df_norm = no_nan_df_norm.drop(column, axis=1)
  pca_vectors_df = pca_vectors_df.drop(column, axis=1)
  

# Adding PCA Noise
def apply_pca_noise(row, pca_vectors_df):
    value = row.copy()
    for _, vector in pca_vectors_df.iterrows():
        mag = vector["ExplainedStd"]
        vector = vector.drop("ExplainedStd")
        vector *= np.random.normal(loc=0, scale=0.5)
        vector *= mag
        value += vector
    
    return value

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Learn a NN that predicts REIT returns given all other data
from sklearn.model_selection import train_test_split
trainingData, testingData = train_test_split(no_nan_df_norm, test_size=0.3, random_state=27)

class MyDataset(Dataset):
    def __init__(self, data, is_training=True):
        self.data = data
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.is_training:
            # Apply augmentation only to training data
            augmented_row = apply_pca_noise(row, pca_vectors_df) 
        else:
            augmented_row = row  # No augmentation during testing

        input_data = augmented_row.drop('REIT')
        output_data = augmented_row['REIT']

        return torch.tensor(input_data.values, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)

trainingDataset = MyDataset(trainingData, is_training=True)
testingDataset = MyDataset(testingData, is_training=False)

from torch.utils.data import DataLoader

batchSize = 100
train_dataloader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(testingDataset, batch_size=batchSize, shuffle=True)

import torch.nn.functional as F
class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    # Define layers
    self.linear1 = nn.Linear(len(no_nan_df_norm.columns) - 1, 3)
    self.linear2 = nn.Linear(3, 1)

  def forward(self, x):
    # Pass data through layers
    out = self.linear1(x)
    out = F.sigmoid(out)
    out = self.linear2(out)
    return out

model = NeuralNetwork()


# Training Loop
import matplotlib.pyplot as plt
device = torch.device("cpu")
def train(network,  data_generator, loss_function, optimizer, logging = 200):
  network.train() 
  num_batches = 0
  avg_loss = 0
  for batch, (input_data, target_output) in enumerate(data_generator):
    input_data, target_output = input_data.to(device), target_output.to(device)
    optimizer.zero_grad()                            
    prediction = network(input_data)      
    target_output = target_output.flatten()
    prediction = prediction.flatten()
    loss = loss_function(prediction, target_output)  
    loss.backward()                                  
    optimizer.step()
    avg_loss += loss.item()
    num_batches += 1
    if ((batch+1)%logging == 0): 
        print('Batch [%d/%d], Train Loss: %.4f' %(batch+1, len(data_generator.dataset)/len(target_output), avg_loss/num_batches))
  return avg_loss/num_batches

def test(network, test_loader, loss_function):
  network.eval()
  test_loss = 0
  num_batches = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      target = target.flatten()
      output = network(data).flatten()
      test_loss += loss_function(output, target).item()
      num_batches += 1
  test_loss /= num_batches
  return test_loss

def logResults(epoch, num_epochs, train_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, print_interval=1000):
  if (epoch%print_interval == 0):  
    print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f' %(epoch+1, num_epochs, train_loss, test_loss))
  train_loss_history.append(train_loss)
  test_loss_history.append(test_loss)
  epoch_counter.append(epoch)

def graphLoss(epoch_counter, train_loss_hist, test_loss_hist, loss_name="Loss", start = 0):
  fig = plt.figure()
  plt.plot(epoch_counter[start:], train_loss_hist[start:], color='blue')
  plt.plot(epoch_counter[start:], test_loss_hist[start:], color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('#Epochs')
  plt.ylabel(loss_name)
  plt.show()
  

def trainAndGraph(network, training_generator, testing_generator, loss_function, optimizer, num_epochs, learning_rate, logging_interval=1):
  #Arrays to store training history
  test_loss_history = []
  epoch_counter = []
  train_loss_history = []
  best_loss = 999999
  for epoch in range(num_epochs):
    avg_loss = train(network, training_generator, loss_function, optimizer)
    test_loss = test(network, testing_generator, loss_function)
    logResults(epoch, num_epochs, avg_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, logging_interval)
    if (test_loss < best_loss):
      best_loss = test_loss
      torch.save(network.state_dict(), 'best_model.pt')
    if (test_loss < 0.2):
        break

  graphLoss(epoch_counter, train_loss_history, test_loss_history)
  
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lossFunction = nn.MSELoss()
trainAndGraph(model, train_dataloader, test_dataloader, lossFunction, optimizer, 500, 0.03)
          
nan_df = df[df.isnull().any(axis=1)].copy()
nan_df.loc[:, '10YrChange'] = df['10YrRate'] - df['10YrRate'].shift(1)
nan_df.loc[0, "10YrChange"] = 0.0027
nan_df_ = nan_df.drop("REIT", axis=1).drop("Year", axis=1)
nan_df_ = nan_df_.apply(lambda x : (x - means.drop("REIT")) / stds.drop("REIT"), axis=1)

for column in drop_columns:
  nan_df_ = nan_df_.drop(column, axis=1)

outputs = []
with torch.no_grad():
    for index, vector in nan_df_.iterrows():
        outputs.append(model(torch.tensor(vector.values, dtype=torch.float32)).item())

for i in range(len(outputs)):
  outputs[i] += np.random.normal(loc=0, scale=0.0) # Give some noise to data so that later models don't learn that this has low variance Y/Y
nan_df_["REIT"] = outputs
nan_df_ = nan_df_.apply(lambda x : (x * stds) + means, axis=1)
nan_df_["Year"] = nan_df["Year"]

for index, vector in nan_df_.iterrows():
    df.loc[index, "REIT"] = vector["REIT"]# * stds["REIT"] + means["REIT"]
    
df = df.round(4)
print(df)
df.to_csv('data//estimatedData.csv', index=False)