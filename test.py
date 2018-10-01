import torch
import torchvision
import ssd_model
from torchsummary import summary

numb_categories = 20+1
k=5
bias = -3.0

#lets see what this looks like
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model1 = torchvision.models.resnet18(pretrained=True)
model1 = torchvision.models.vgg16(pretrained=True)
# print(model1)
#show the layer sizes, given an input
model1.to(device)
summary(model1, (3, 224, 224))
# MaxPool2d-31            [-1, 512, 7, 7]               0  batch size by 512 depth by 7*7 =>512*7*7= 25088

print(model1.classifier[0].in_features)
print(model1.classifier[0].out_features)


for name, child in model1.named_children():
    for name2, params in child.named_parameters():
        print(name, name2)

#get a list of the layers of interest
# modules=list(model1.children())[:-2]  #strip the last 2 layers for resnet
modules=list(model1.children())[:-1]  #strip the last  layer for vgg16
print (modules)

# modules_dict = {name: module for (name,module) in model1.named_children()}.pop("classifier",None)
modules_dict = {name: module for (name,module) in model1.named_children()}
print (modules_dict)
print("*"*20)
print(model1.classifier[0])
print("*"*20)
for key,value in modules_dict.items():
    print(key)
print (model1.features[0])
print (model1.features[1])
print("*"*20)
print (model1.features)
print (type(model1.features))
#create a new head
ssd_head = ssd_model.SSD_MultiHead(k, bias, numb_categories)

#add to model
modules.append(ssd_head)
1
#make into a model
model1 = torch.nn.Sequential(*modules) #recreate (lose the names of the layers though)
model1.to(device)

#show the layer contents
print(model1)

#show the layer sizes, given an input
summary(model1, (3, 224, 224))

#freeze all
for param in model1.parameters():
    param.requires_grad = False

#what is the output size of the last conv layer?
num_linear_inputs = model1[0].in_features


ssd_head = ssd_model.SSD_MultiHead(k, bias, numb_categories)

from torchsummary import summary


# num_outputs = 12 # number of weedlings
# model_conv.fc = nn.Linear(num_linear_inputs, num_outputs)
# model_conv = model_conv.to(device)
