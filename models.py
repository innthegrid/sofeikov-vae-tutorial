import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self, num_hidden=8):
    super().__init__()

    # Set the number of hidden units - bottleneck size - from parameter
    self.num_hidden = num_hidden

    # Define the encoder: compress 784 to num_hidden
    # Sequential pipeline - data flows through layers in order
    self.encoder = nn.Sequential(
      nn.Linear(784, 256), # Input size: 784, Output size: 256
      nn.ReLU(), # ReLU activation function
      nn.Linear(256, self.num_hidden),
      nn.ReLU(),
    )

    # Define the decoder: reconstruct num_hidden back to 784
    self.decoder = nn.Sequential(
      nn.Linear(self.num_hidden, 256),
      nn.ReLU(),
      nn.Linear(256, 784),
      nn.Sigmoid(), # Sigmoid activation function, compresses the output to (0,1) range
    )

  def forward(self, x):
    # Pass the input through the encoder
    encoded = self.encoder(x)

    # Pass the encoded representation through the decoder
    decoded = self.decoder(encoded)

    # Return the encoded representation and the reconstructed output
    return encoded, decoded