import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

with open('regression_train.txt', 'r') as f:
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip()
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
    f.close()

np.random.seed(0)
x = np.array(x).reshape(-1, 1)


from sklearn.preprocessing import PolynomialFeatures
#add polynomial features
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

# Fit the model
model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

# Plot the data and the model
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')

plt.xlabel('x')
plt.ylabel('y')


plt.show()

with open('regression_test.txt', 'r') as f:
    lines = f.readlines()
    x_test = []
    y_test = []
    for line in lines:
        line = line.strip()
        x_test.append(float(line.split()[0]))
        y_test.append(float(line.split()[1]))
    f.close()

#calculate the mse 

from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(y, y_pred)
#print("Mean Squared Error:", mse)



x_test = np.array(x_test).reshape(-1, 1)
#map x to polynomial features
x_poly = poly.transform(x_test)
y_pred = model.predict(x_poly)

# Plot the data and the model
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

#calculate the mse
mse = mean_squared_error(y_test, y_pred)
print("Linear regression Mean Squared Error on test set:", mse)

np.random.seed(0)
torch.manual_seed(0)
# Load the data
with open('regression_train.txt', 'r') as f:
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip()
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
    f.close()

datax = np.array(x).reshape(-1, 1)
y = np.array(y)

# Standardize the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(datax)


x_poly = x_scaled
# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_poly, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define the model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.degree = 3
        self.hidden1 = nn.Linear(1*self.degree, 104)
        self.hidden2 = nn.Linear(104, 184)
        self.hidden3 = nn.Linear(184, 208)
        #self.dropout3 = nn.Dropout(0.227971273850112)
        #self.hidden4 = nn.Linear(160*3, 184*3)
        #self.dropout4 = nn.Dropout(0.203282814201168)
        
        self.activate = nn.LeakyReLU()
        self.output = nn.Linear(208, 1)

    def forward(self, x):
        
        x = [x]
        for i in range(2, self.degree+1):
            x.append(torch.pow(x[0], i)) 
        x = torch.cat(x, dim=1)
        x = self.activate(self.hidden1(x))
        x = self.activate(self.hidden2(x))
        x = self.activate(self.hidden3(x))
        x = self.output(x)
        return x

# k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=0)

fold_mse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x_tensor)):
    print(f"Fold {fold + 1}/{k}")
    
    # Split data into training and validation sets
    x_train_fold = x_tensor[train_idx]
    y_train_fold = y_tensor[train_idx]
    x_val_fold = x_tensor[val_idx]
    y_val_fold = y_tensor[val_idx]
    
    # Create a new model for each fold
    model = NeuralNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003188384254259)
    
    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train_fold)
        loss = criterion(outputs, y_train_fold)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate the model
    model.eval()
    with torch.no_grad():
        val_predictions = model(x_val_fold)
        mse = mean_squared_error(y_val_fold.numpy(), val_predictions.numpy())
        fold_mse.append(mse)
        print(f"Fold {fold + 1} MSE: {mse:.4f}")

# Cross-validation results
mean_mse = np.mean(fold_mse)
std_mse = np.std(fold_mse)
#print(f"\nCross-Validation Mean MSE: {mean_mse:.4f}, Std: {std_mse:.4f}")

# Train final model
model = NeuralNet()
optimizer = optim.Adam(model.parameters(), lr= 0.003188384254259)
critertion = nn.MSELoss()
num_epochs = 1000
# Train the final model on the full dataset
for epoch in range(num_epochs):
    model.train()
    outputs = model(x_tensor)
    loss = critertion(outputs, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict and plot
with torch.no_grad():
    predictions = model(x_tensor).numpy()

plt.scatter(datax, y, label="Actual Data")
plt.plot(datax, predictions, color='red', label="Model Predictions")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Neural Network Final Model Fit')
plt.show()


#test data
with open ('regression_test.txt', 'r') as f:
    lines = f.readlines()
    x_test = []
    y_test = []
    for line in lines:
        line = line.strip()
        x_test.append(float(line.split()[0]))
        y_test.append(float(line.split()[1]))
    f.close()

x_test = np.array(x_test).reshape(-1, 1)

x_test_scaled = scaler.transform(x_test)

#X_test_poly = poly.transform(x_test)
X_test_poly = x_test_scaled
X_test_poly = torch.tensor(X_test_poly, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_poly)

#convert the predictions to numpy array
y_pred = y_pred.numpy()


# Plot the data and the model
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, color='red')

plt.xlabel('x')
plt.ylabel('y')

plt.show()


# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)

print('Neural Network: Mean Squared Error on test set:', mse)



import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Load the data
np.random.seed(0)
with open('regression_train.txt', 'r') as f:
    lines = f.readlines()
    x = []
    y = []
    for line in lines:
        line = line.strip()
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))
    f.close()

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# Define Bayesian Neural Network model with PyMC
num_samples = 100000
model = pm.Model()

with model:
    # Defining our priors
    w0 = pm.Normal('w0', mu=0, sigma=20)
    w1 = pm.Normal('w1', mu=0, sigma=20)
    w2 = pm.Normal('w2', mu=0, sigma=20)
    w3 = pm.Normal('w3', mu=0, sigma=20)
    w4 = pm.Uniform('w4', lower=0, upper=200)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + w1 * x + w2 * x**2 + w3 * x**3 + w4 # auxiliary variables

    likelihood = pm.Normal('y', mu=y_est, sigma=sigma, observed=y)
    
    # Inference
    sampler = pm.NUTS()  # Hamiltonian MCMC with No U-Turn Sampler
    idata = pm.sample(num_samples, tune=10000, step=sampler, progressbar=True, cores=4)

az.plot_trace(idata, combined=True)
az.plot_trace(idata, legend=True)   
plt.show()