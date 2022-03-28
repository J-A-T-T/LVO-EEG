import numpy as np
import torch
from models.eegnet1 import EEGNet
import matplotlib.pyplot as plt
from utils.plot_eeg import plot_eeg
import cv2

example = np.random.rand(1,1,5000,4) 
example = torch.FloatTensor(example)

model = EEGNet(output = 1)

pred = model(example)
pred.backward()

gradients = model.get_activations_gradient() # 1 x 4 x 2 x 312

pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # size 4

activations = model.get_activations(example).detach()

# weight the channels by corresponding gradients
for i in range(activations.size(1)):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
# expression (2) in https://arxiv.org/pdf/1610.02391.pdf
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.squeeze())

heatmap = heatmap.numpy() # 2 x 312

example = torch.squeeze(example).cpu().detach().numpy()

heatmap = cv2.resize(heatmap, (example.shape[1], example.shape[0]))
heatmap = np.uint8(255 * heatmap)
superimposed_img = heatmap * 0.4 + example
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
plot_eeg(heatmap)
# plot_eeg(example)
# plot_eeg(superimposed_img)
plt.show()