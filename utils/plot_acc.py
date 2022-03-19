import matplotlib.pyplot as plt

def plot_acc_loss(train_accs, test_accs, train_losses, test_losses, caption):
  # Visualize the loss / acc

  plt.plot(train_accs, label="Train")
  plt.plot(test_accs, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.title("Classification Accuracy vs Epoch: {}".format(caption))
  plt.figure()

  # Visualize the loss / acc
 
  plt.plot(train_losses, label="Train")
  plt.plot(test_losses, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Cross Entropy Loss")
  plt.legend()
  plt.title("Cross Entropy Loss vs Epoch: {}".format(caption))
  plt.figure()

  return 