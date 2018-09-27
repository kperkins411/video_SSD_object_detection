import torch
import torchvision
import ssd_model

model_conv = torchvision.models.resnet34(pretrained=True)

#freeze all
for param in model_conv.parameters():
    param.requires_grad = False

#what is the output size of the last conv layer?
num_linear_inputs = model_conv.fc.in_features

numb_categories = 20+1
k=1
bias = -3.0

ssd_head = ssd_model.SSD_MultiHead(k, bias, numb_categories)



# num_outputs = 12 # number of weedlings
# model_conv.fc = nn.Linear(num_linear_inputs, num_outputs)
# model_conv = model_conv.to(device)
