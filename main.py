import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import download_mnist, show_images, show_reconstructions
from models import VAE

def vae_loss_function(recon_x, x, mu, log_var):
  # Compute the reconstruction loss (binary cross-entropy)
  BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

  # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
  KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

  return BCE + KLD

def main():
  # Hyperparameters
  learning_rate = 0.001
  batch_size = 64
  num_epochs = 15
  # VAE Add
  hidden_dim = 20 # same as num_hidden, changed from 8 to 20

  # Data preparation
  X_train, y_train, X_test, y_test = download_mnist()

  # Convert the training data to PyTorch tensors
  # Create a DataLoader to handle batching
  train_loader = DataLoader(
    torch.from_numpy(X_train),
    batch_size=batch_size,
    shuffle=True
  )

  # Model initialization
  # Create the autoencoder model and optimizer
  # model = AutoEncoder() - Original for Autoencoder implementation
  # VAE Add
  model = VAE(num_hidden=hidden_dim)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Define the loss function (Mean Squared Error)
  # criterion = nn.MSELoss() - Original for Autencoder implementation
  # For VAE, we defined a custom loss function above

  # Training loop
  print("Start training...")
  for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
      # Forward pass
      # encoded, decoded = model(batch) - Original for Autoencoder implementation
      # VAE Add
      _, decoded, mu, log_var = model(batch)

      # Compute the loss - comparing reconstruction to original
      # loss = criterion(decoded, batch) - Original for Autoencoder implementation
      # VAE Add
      loss = vae_loss_function(decoded, batch, mu, log_var)

      # Backpropagation
      optimizer.zero_grad() # Clear previous gradients
      loss.backward() # Compute gradients
      optimizer.step() # Update weights

      # Update the running loss
      # total_loss += loss.item() * batch.size(0) - Original for Autoencoder implementation
      # VAE Add - don't need to multiply by batch.size(0) since loss is already summed in loss function
      total_loss += loss.item()

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss:.4f}")

  # Visualization (post-training)
  print("\nVisualizing Reconstruction...")
  model.eval() # Set the model to evaluation mode
  with torch.no_grad(): # Disable gradient calcuation
    # Get a few test images
    test_input = torch.from_numpy(X_test[:5])
    # _, reconstructed = model(test_input) - Original for Autoencoder implementation
    # VAE Add - return more values
    _, reconstructed, _, _ = model(test_input)

    # Display original vs reconstructed images
    # print("Originals (Top) vs Reconstructions (Bottom):") - Original for Autoencoder implementation
    # VAE Add
    print("\nShowing reconstructions (original vs VAE output):")
    show_reconstructions(X_test[:5], reconstructed.numpy(), y_test[:5])

    # VAE Add - Show generated samples
    print("\nShowing generated samples:")
    generated = model.sample(5)
    show_images(generated.numpy(), ["Gen"]*5)


if __name__ == "__main__":
  main()