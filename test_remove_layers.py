import torch
import torchvision
import ssd_model
from torchsummary import summary

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



model1 = torchvision.models.resnet18(pretrained=False)

modules=list(model1.children())[:-2]  #strip the last 2 layers
model1 = torch.nn.Sequential(*modules) #recreate (lose the names of the layers though)


f = open("res_minus2","w")
f.write(model1.__str__())
f.close()

print(model)
model.fc = Identity()
print(model)

model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)