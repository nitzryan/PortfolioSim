import pandas as pd
from scipy.stats import zscore

# Setup Data
df = pd.read_csv('data//estimatedData.csv')
df = df.drop("Year", axis=1)
df.loc[:, '10YrChange'] = df['10YrRate'] - df['10YrRate'].shift(1)
df.loc[0, '10YrChange'] = 0.0027

# Modify column order to control order of prediction
# Try to predict general conditions first, than predict asset returns
varList = ["USGDP%", "Inflation$", "10YrChange", "10YrRate", "3Month", "10Yr", "Baa Corporate", "SP500", "Real Estate", "REIT", "Gold", "PE", "CAPE"]
df = df[varList]
norm_df = df.apply(zscore)
means = df.mean()
stds = df.std()

# Generate PCA Vectors
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=len(means))
pca.fit_transform(norm_df)
pca_vectors_df = pd.DataFrame(pca.components_, columns=norm_df.columns)
pca_vectors_df = pca_vectors_df.assign(ExplainedStd=np.sqrt(pca.explained_variance_ratio_))
pca_vectors_df["ExplainedStd"] = pca_vectors_df["ExplainedStd"] / pca_vectors_df["ExplainedStd"].sum()

# Adding PCA Noise
def apply_pca_noise(row, pca_vectors_df, noiseScale):
    value = row.copy()
    for _, vector in pca_vectors_df.iterrows():
        mag = vector["ExplainedStd"]
        vector = vector.drop("ExplainedStd")
        vector *= np.random.normal(loc=0, scale=noiseScale)
        vector *= mag
        value += vector
    
    return value

''' 
General Strategy: Predict variable distribution for a single variable
Then, in next model, select values into that distribution to generate the next distribution
For N variables, this results in building N models, each only predicting one distribution
Each model's distribution takes in a value from the previous model's distributions, not the distribution itself
'''
# Distribution will be modeled by turning to classification with buckets
import torch
# cutoffs = [-3.5, -3, -2.666, -2.333, -2, -1.8, -1.6, -1.4, -1.2, -1,
#                -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0,
#                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
#                1.2, 1.4, 1.6, 1.8, 2, 2.333, 2.666, 3, 3.5]
cutoffs = [-3.5, -3, -2.5, -2, -1.66, -1.33, -1, -0.75, -0.5, -0.25, 0,
           0.25, 0.5, 0.75, 1, 1.33, 1.66, 2, 2.5, 3, 3.5
]
# cutoffs = [-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
numBuckets = len(cutoffs) + 1
def ZScoreToBucket(score, smooth_class=False):
    buckets = torch.zeros(len(cutoffs) + 1, dtype=torch.float32)
    for i in range(len(cutoffs)):
        if score < cutoffs[i]:
            buckets[i] = 1
            # if smooth_class:
            #   offsets = [0.033, 0.066, 0.1]
            #   if (i >= 3):
            #     buckets[i] -= offsets[0]
            #     buckets[i - 3] = offsets[0]
            #   if (i >= 2):
            #     buckets[i] -= offsets[1]
            #     buckets[i - 2] = offsets[1]
            #   if (i >= 1):
            #     buckets[i] -= offsets[2]
            #     buckets[i - 1] = offsets[2]
            #   if (i < numBuckets - 3):
            #     buckets[i] -= offsets[0]
            #     buckets[i + 3] = offsets[0]
            #   if (i < numBuckets - 2):
            #     buckets[i] -= offsets[1]
            #     buckets[i + 2] = offsets[1]
            #   if (i < numBuckets - 1):
            #     buckets[i] -= offsets[2]
            #     buckets[i + 1] = offsets[2]
            return buckets
    
    buckets[len(cutoffs)] = 1
    return buckets

def BucketToZScore(bucket):
    if bucket == 0:
        lowBand = -4
    else:
        lowBand = cutoffs[bucket - 1]
    if bucket == len(cutoffs):
        highBand = 4
    else:
        highBand = cutoffs[bucket]
    return np.random.uniform(low=lowBand, high=highBand)

def DistributionToZScore(dist):
    r = np.random.uniform(low=0, high=1)
    for i in range(dist.size(dim=0)):
        r -= dist[i]
        if r <= 0:
            return BucketToZScore(i)
    
    print("This should not be reached")
    return BucketToZScore(dist.size(dim=0))

# Generate Test/Train Split on years
indices = range(norm_df.shape[0] - 1)
from sklearn.model_selection import train_test_split
trainIndices, testIndices = train_test_split(indices, test_size=0.15, random_state=570)
print([x + 1929 for x in testIndices])

# Generate input, output based on index of variable
# For datapoint N, variable M, input is certain variables of point N, N-1, 
# And certain variables 0...M-1 for point N
# Output is is M of point N
from torch.utils.data import Dataset

# Dependencies: ([prev], [current])
varDependencies = [
  ([0,1,2,3,4,8,11,12],[]), #GDP
  ([0,1,2,3,4,6,8,10,11,12],[0]), #Inflation
  ([2,3,8,10],[0,1]), #10 Year Change
  ([3],[2]), #10 Year Rate
  ([0,1,3,4],[0,1,3]), #3 Month Returns
  ([3,4],[2]), #10 Year Returns
  ([3,4,8,10,12],[0,1,2,5]), #Baa Corporate Returns
  ([0,1,7,11,12],[0,1,6]), #SP500 Returns (Most of these are really low correlations and heavily biased.  Only Current Baa Corp and prev CAPE are abs(corr)>=0.2)
  ([0,1,8,9],[0,1,2,5]), #Real Estate Returns
  ([3,12],[0,2,3,6,7,8]), #REIT Returns
  ([0,2,7,8,10,12],[0,1,2,6,7,8]), #Gold Returns
  ([0,1,2,3,4,8,11,12],[0,1,3,4,7,8,9]), #PE Ratio
  ([0,1,3,4,10,11,12],[0,1,3,4,7,11]) # CAPE Ratio
]
["USGDP%", "Inflation$", "10YrChange", "10YrRate", "3Month", "10Yr", "Baa Corporate", "SP500", "Real Estate", "REIT", "Gold", "PE", "CAPE"]
input_sizes = []
for dep in varDependencies:
  input_sizes.append(len(dep[0]) + len(dep[1]))

import torch.nn as nn
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.isTraining = True
        self.index = 0
    
    def __len__(self):
        print(len(self.data))
        return len(self.data)
    
    def setIndex(self, index):
        self.index = index
        
    def setTraining(self, isTraining):
        self.isTraining = isTraining
        
    def __getitem__(self, idx):
        prev = self.data.iloc[idx]
        cur = self.data.iloc[idx + 1]
        
        # if self.isTraining:
        #     prev = apply_pca_noise(prev, pca_vectors_df, 1)
        #     cur = apply_pca_noise(cur, pca_vectors_df, 1)
            
        inputData = torch.zeros(input_sizes[self.index], dtype=torch.float32)
        n = 0
        for prevDep in varDependencies[self.index][0]:
          inputData[n] = prev.iloc[prevDep]
          n += 1
        for curDep in varDependencies[self.index][1]:
          inputData[n] = cur.iloc[curDep]
            
        outputData = ZScoreToBucket(cur.iloc[self.index], smooth_class=self.isTraining)
        return inputData, outputData
    
dataset = CustomDataset(norm_df)
class MySubsetSampler(torch.utils.data.Sampler):    
    def __init__(self, indices):        
        self.indices = indices    
    def __iter__(self):        
        return (self.indices[i] for i in range(len(self.indices)))    
    def __len__(self):        
        return len(self.indices)

trainLoader = torch.utils.data.DataLoader(
    dataset, batch_size=16,
    sampler= MySubsetSampler(trainIndices)
)
testLoader = torch.utils.data.DataLoader(
    dataset, batch_size=1,
    sampler= MySubsetSampler(testIndices)
)

# Network
# Narrows to learn environment, then uses that to fill buckets
import torch.nn.functional as F
class NeuralNetwork(nn.Module):
  def __init__(self, inputSize):
    super(NeuralNetwork, self).__init__()
    # Define layers
    self.linear1 = nn.Linear(inputSize, 3)
    self.linear2 = nn.Linear(3, numBuckets)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.linear1(x)
    out = F.sigmoid(out)
    out = self.linear2(out)
    return out

# Training
device = torch.device("cpu")
def train(network,  data_generator, loss_function, optimizer, scheduler, logging = 200):
  data_generator.dataset.setTraining(True)
  network.train() 
  num_batches = 0
  avg_loss = 0
  for batch, (input_data, target_output) in enumerate(data_generator):
    input_data, target_output = input_data.to(device), target_output.to(device)
    optimizer.zero_grad()                            
    prediction = network(input_data)      
    target_output = target_output
    prediction = prediction
    loss = loss_function(prediction, target_output)  
    loss.backward()                                  
    optimizer.step()
    avg_loss += loss.item()
    num_batches += 1
    if ((batch+1)%logging == 0): 
        print('Batch [%d/%d], Train Loss: %.4f' %(batch+1, len(data_generator.dataset)/len(target_output), avg_loss/num_batches))
  
  scheduler.step()
  return avg_loss/num_batches

softmax = nn.Softmax(dim=1)
def test(network, data_generator, loss_function):
  data_generator.dataset.setTraining(False)
  network.eval()
  test_loss = 0
  num_batches = 0
  with torch.no_grad():
    for batch, (data, target) in enumerate(data_generator):
      data, target = data.to(device), target.to(device)
      output = network(data)
      test_loss += loss_function(output, target).item()
      num_batches += 1
  test_loss /= num_batches
  return test_loss

def logResults(epoch, num_epochs, train_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, print_interval=1000):
  # if (epoch%print_interval == 0):  
  #   print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f' %(epoch+1, num_epochs, train_loss, test_loss))
  train_loss_history.append(train_loss)
  test_loss_history.append(test_loss)
  epoch_counter.append(epoch)
  
import matplotlib.pyplot as plt
def graphLoss(epoch_counter, train_loss_hist, test_loss_hist, loss_name="Loss", start = 0):
  plt.plot(epoch_counter[start:], train_loss_hist[start:], color='blue')
  plt.plot(epoch_counter[start:], test_loss_hist[start:], color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('#Epochs')
  plt.ylabel(loss_name)
  plt.show()
  
def trainAndGraph(network, trainGenerator, testGenerator, loss_function, optimizer, scheduler, num_epochs, logging_interval=1):
  test_loss_history = []
  epoch_counter = []
  train_loss_history = []
  best_loss = 99999
  best_epoch = -1
  for epoch in range(num_epochs):
    avg_loss = train(network, trainGenerator, loss_function, optimizer, scheduler)
    test_loss = test(network, testGenerator, loss_function)
    logResults(epoch, num_epochs, avg_loss, train_loss_history, test_loss, test_loss_history, epoch_counter, logging_interval)
    if (test_loss < best_loss):
      best_loss = test_loss
      best_epoch = epoch
      #torch.save(network.state_dict(), 'best_model.pt')
    if epoch > 20 and test_loss > (1.1 * best_loss):
      print(f"Best Loss: {best_loss:4f} on Epoch {epoch + 1}")
      break
    elif epoch == num_epochs - 1:
      print(f"Best Loss: {best_loss:4f} on Epoch {best_epoch + 1} (did not stop early)")
      break
  
  #graphLoss(epoch_counter, train_loss_history, test_loss_history)
  return best_loss
  
# Loop through variables
models = []
for index in range(len(varList)):
    print("Variable " + varList[index])
    dataset.setIndex(index)
    #model = NeuralNetwork(1 * len(varList) + index)
    model = NeuralNetwork(len(varDependencies[index][0]) + len(varDependencies[index][1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9, last_epoch=-1)
    lossFunction = nn.CrossEntropyLoss()
    lastLoss = trainAndGraph(model, trainLoader, testLoader, lossFunction, optimizer, scheduler, 1000, 10)
    models.append(model)
    #print(f"Total Loss: {totalBestLoss:4f}, Best Loss: {bestLoss:4f}")
    #models.append(model)
    
# Use Models to generate possible scenarioes
import warnings
with torch.no_grad():
  for scenario in range(0, 1000): #Scenarios
      lastData = norm_df.iloc[-1,:]
      scenario_df = pd.DataFrame(columns=varList)
      total_delta = np.zeros_like(lastData) # Pulls towards historical returns
      for year in range(100):
          thisData = []
          for var in range(len(varList)):
            # Generate Input Vector
            input = torch.zeros(input_sizes[var], dtype=torch.float32)
            n = 0
            for prev in varDependencies[var][0]:
              input[n] = lastData.iloc[prev]
              n += 1
            for cur in varDependencies[var][1]:
              input[n] = thisData[cur]
              n += 1
            
            output = F.softmax(models[var](input), dim=0)
              
            value = DistributionToZScore(output)
            if year != 0:
              value -= total_delta[var] * 0.05
            total_delta[var] += value
            thisData.append(value)
              
          s = pd.Series(thisData, index=scenario_df.columns)
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scenario_df = pd.concat([scenario_df, s.to_frame().T], ignore_index=True)
            
          lastData = scenario_df.iloc[-1,:]
        
      scenario_df = scenario_df.apply(lambda x : (x * stds) + means, axis=1)
      scenario_df["Year"] = scenario_df.index + 2024
      # Get the existing column names
      cols = scenario_df.columns.tolist()

      # Reorder the columns with 'Year' first
      cols = ['Year'] + cols[:-1]  

      # Assign the reordered columns to the dataframe 
      scenario_df = scenario_df[cols]
      year_col = scenario_df.pop("Year")
      scenario_df.insert(0, 'Year', year_col)
      scenario_df = scenario_df.round(4)
      scenario_df.to_csv(f"data//scenariosSlowMerge//scenario{scenario + 1}.csv", index=False)