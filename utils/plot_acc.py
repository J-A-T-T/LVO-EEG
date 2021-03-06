import matplotlib.pyplot as plt

def plot_acc(train_accs, test_accs, caption):
  # Visualize the loss / acc

  plt.plot(train_accs, label="Train")
  plt.plot(test_accs, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.title("Classification Accuracy vs Epoch: {}".format(caption))
  plt.show()

  return 