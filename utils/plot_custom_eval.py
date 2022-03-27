import matplotlib.pyplot as plt

def plot_custom_eval(train_eval, test_eval, caption):
  # Visualize the loss / acc

  plt.plot(train_eval, label="Train")
  plt.plot(test_eval, label="Test")
  plt.xlabel("Epoch")
  plt.ylabel("Custom Evaluation")
  plt.legend()
  plt.title("Classification Accuracy vs Epoch: {}".format(caption))
  plt.show()

  return 