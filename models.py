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
  
class VAE(AutoEncoder):
  def __init__(self, num_hidden=8):
    super().__init__(num_hidden) # Calls AutoEncoder

    # Add mu and log_var layers for reparameterization
    self.mu = nn.Linear(self.num_hidden, self.num_hidden)
    self.log_var = nn.Linear(self.num_hidden, self.num_hidden)

  def reparameterize(self, mu, log_var):
    # Compute the standard deviation from the log variance
    std = torch.exp(0.5 * log_var)

    # Generate random noise (normal distribution) using the same shape as std
    eps = torch.randn_like(std)

    # Return the reparameterized sampel
    return mu + eps * std
  
  def forward(self, x):
    # Pass the input through the encoder
    encoded = self.encoder(x)

    # Compute the mean and log variance veectors
    mu = self.mu(encoded)
    log_var = self.log_var(encoded)

    # Reparameterize the latent variable
    z = self.reparameterize(mu, log_var)

    # Pass the latent variable through the decoder
    decoded = self.decoder(z)
    
    # Return the encoded output, decoded output, mean, and log variance
    return encoded, decoded, mu, log_var
  
  def sample(self, num_samples):
    with torch.no_grad():
      # Generate random noise with normal distribution N(0, 1)
      z = torch.randn(num_samples, self.num_hidden)
      samples = self.decoder(z)
    return samples