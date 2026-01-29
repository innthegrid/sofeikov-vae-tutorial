import os
import gzip
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

def download_mnist():
  """
  Download MNIST dataset
  Return (X_train, y_train, X_test, y_test)
  """

  url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
  filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
  ]

  data = []
  for filename in filenames:
    if not os.path.exists(filename):
      print(f"Downloading {filename}...")
      urllib.request.urlretrieve(url + filename, filename)

    with gzip.open(filename, 'rb') as f:
      if 'labels' in filename:
        # Labels offset is 8 bytes
        data.append(np.frombuffer(f.read(), np.uint8, offset=8))
      else:
        # Images offset is 16 bytes
        # Reshape to flattened 784 (28x28)
        data.append(np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784))

  X_train, y_train, X_test, y_test = data

  # Normalize pixel values to [0, 1]
  X_train = X_train.astype(np.float32) / 255.0
  X_test = X_test.astype(np.float32) / 255.0

  # Convert label types
  y_train = y_train.astype(np.int64)
  y_test = y_test.astype(np.int64)

  return X_train, y_train, X_test, y_test

def show_images(images, labels, num_to_show=5):
  """
  Display a row of images and their labels
  """
  # Select a subset of images and labels
  images = images[:num_to_show]
  labels = labels[:num_to_show]

  # Unflatten the data
  pixels = images.reshape(-1, 28, 28)
  fig, axs = plt.subplots(1, len(images), figsize=(12, 3))

  # Handle case when only 1 image is shown
  if len(images) == 1:
    axs = [axs]

  for i in range(len(images)):
    axs[i].imshow(pixels[i], cmap='gray')
    axs[i].set_title(f"Label: {labels[i]}")
    axs[i].axis('off')

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  X_train, y_train, X_test, y_test = download_mnist()
  print(f"Loaded {len(X_train)} training images.")
  show_images(X_train, y_train)