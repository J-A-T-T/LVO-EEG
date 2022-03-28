
# Author: Vinicius Arruda
# viniciusarruda.github.io
# Source code modified from: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

# In the current directory create a folder called 'data' and inside it another folder called 'Elephant' 
# and inside, paste the image you want to check the GradCAM

torch.manual_seed(1)

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((32, 32)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='./data2/', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        
        self.fc1 = nn.Linear(3*32*32, 2*16*16, bias=False)
        self.fc2 = nn.Linear(2*16*16, 2*2*2, bias=False)
        self.fc3 = nn.Linear(2*2*2, 2, bias=False)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        
        x = x.view(1, 3*32*32)    
        x = self.fc1(x) 
        x = self.fc2(x)
        x = x.view(1, 2, 2, 2)    
        return x

    def get_activations_before(self, x):
        
        x = x.view(1, 3*32*32)    
        x = self.fc1(x)
        x = x.view(1, 2, 16, 16)     
        return x
        
    def forward(self, x):
        x = x.view(1, 3*32*32)    # 1 x 3072
        
        x = self.fc1(x) # 1 x 512
        x = self.fc2(x) # 1 x 8
        x = x.view(1, 2, 2, 2) # 1 x 2 x 2 x 2   
        x.register_hook(self.activations_hook)

        x = x.view(1, 2*2*2)    
        x = self.fc3(x)

        return x


sn = SimpleNet()

#####
# Training procedure to be placed here
#####

sn.eval()
img, _ = next(iter(dataloader)) # 1 x 3 x 255 x 255
pred = sn(img)

pred_idx = pred.argmax(dim=1)

pred[:, pred_idx].backward()
gradients = sn.get_activations_gradient() # size 1 x 2 x 2 x 2 (reshape to use in torch.mean)

# # pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]) # size 2


# # get the activations of the last convolutional layer
activations = sn.get_activations(img).detach()


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

heatmap = heatmap.numpy()

plt.show()
img = cv2.imread('./data2/Elephant/1.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./map.jpg', superimposed_img)

plt.show()