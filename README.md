# Stock-Price-Prediction

## AIM

To develop a **Recurrent Neural Network (RNN)** model for predicting stock prices using historical data.

---

## Problem Statement and Dataset

Stock price prediction is a challenging task due to the non-linear and volatile nature of financial markets. Traditional methods often fail to capture complex temporal dependencies. Deep learning, specifically **Recurrent Neural Networks (RNNs)**, can effectively model time-series dependencies, making them suitable for stock price forecasting.

* **Problem Statement**:
  Build an RNN model to predict the future stock price based on past stock price data.

* **Dataset**:
  A stock market dataset containing **historical daily closing prices** (e.g., Google, Apple, Tesla, or NSE/BSE data).
  The dataset is usually divided into **training and testing sets** after applying normalization and sequence generation.

---

## Design Steps

### Step 1:

Import required libraries such as `torch`, `torch.nn`, `torch.optim`, `numpy`, `pandas`, and `matplotlib`.

### Step 2:

Load the dataset (e.g., stock closing prices from CSV), preprocess it by **normalizing** values between 0 and 1, and create input sequences for training/testing.

### Step 3:

Define the **RNN model architecture** with an input layer, hidden layers, and an output layer to predict stock prices.

### Step 4:

Compile the model using **MSELoss** as the loss function and **Adam optimizer**.

### Step 5:

Train the model on the training data, recording training losses for each epoch.

### Step 6:

Test the trained model on unseen data and visualize results by plotting the **true stock prices vs. predicted stock prices**.



## Program
#### Name:AADITHYAN R
#### Register Number:212222230001
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(1, 64, 2, batch_first = True)
    self.fc = nn.Linear(64, 1)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out





model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")







```

## Output

### True Stock Price, Predicted Stock Price vs time
<img width="591" height="408" alt="Screenshot (10)" src="https://github.com/user-attachments/assets/9688e123-ee36-4137-924b-58b0718c3031" />



### Predictions 
<img width="324" height="43" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/eb4d8594-b0ec-4afc-9567-6aefec04c1ef" />
<img width="689" height="431" alt="Screenshot (11)" src="https://github.com/user-attachments/assets/51193766-40df-4fd6-9425-cd2e90a245fe" />


## Result
The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.


