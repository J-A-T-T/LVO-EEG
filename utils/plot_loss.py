import matplotlib.pyplot as plt

def plot_loss(train_losses, test_losses, caption):
  # Visualize the loss / acc
 
  plt.plot(train_losses, label="Train")
  plt.plot(test_losses, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Cross Entropy Loss")
  plt.legend()
  plt.title("Cross Entropy Loss vs Epoch: {}".format(caption))
  plt.show()

  return 