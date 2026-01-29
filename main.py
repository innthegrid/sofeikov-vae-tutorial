import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import download_mnist, show_images, show_reconstructions
from models import AutoEncoder

def main():
  # Hyperparameters
  learning_rate = 0.001
  batch_size = 64
  num_epochs = 10

  # Data preparation
  X_train, y_train, X_test, y_test = download_mnist()

  # Convert the training data to PyTorch tensors
  X_train_tensor = torch.from_numpy(X_train)

  # Create a DataLoader to handle batching
  train_loader = DataLoader(
    X_train_tensor,
    batch_size=batch_size,
    shuffle=True
  )

  # Model initialization
  # Create the autoencoder model and optimizer
  model = AutoEncoder()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # Define the loss function (Mean Squared Error)
  criterion = nn.MSELoss()

  # Training loop
  print("Start training...")
  for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
      # Forward pass
      encoded, decoded = model(batch)

      # Compute the loss - comparing reconstruction to original
      loss = criterion(decoded, batch)

      # Backpropagation
      optimizer.zero_grad() # Clear previous gradients
      loss.backward() # Compute gradients
      optimizer.step() # Update weights

      # Update the running loss
      total_loss += loss.item() * batch.size(0)

    # Print the epoch loss
    epoch_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss={epoch_loss:.4f}")

  # Visualization (post-training)
  print("\nVisualizing Reconstruction...")
  model.eval() # Set the model to evaluation mode
  with torch.no_grad(): # Disable gradient calcuation
    # Get a few test images
    test_input = torch.from_numpy(X_test[:5])
    _, reconstructed = model(test_input)

    # Display original vs reconstructed images
    print("Originals (Top) vs Reconstructions (Bottom):")
    show_reconstructions(X_test[:5], reconstructed.numpy(), y_test[:5])

if __name__ == "__main__":
  main()