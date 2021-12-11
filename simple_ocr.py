#
# You can modify this files
#

#
# You can modify this files
#
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision
from PIL import Image

# #mô hình deeplearing đc viết dựa trên pytorch
# #có những phép cnn, fully connected để huấn luyện mô hình.
def resnet_18(class_names):
    network = torchvision.models.resnet18(pretrained=True)
    for param in network.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, len(class_names))
    return network


# class Net(nn.Module):
#     def __init__(self, class_names):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 12, 5)
#         self.conv3 = nn.Conv2d(12, 32, 3)
#         self.fc1 = nn.Linear(20000, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, len(class_names))

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class HoadonOCR:
    #khởi tạo các biến số
    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'others', 'phuclong', 'starbucks']
        self.class_names = {'highlands': 0, 'starbucks': 3, 'phuclong': 2, 'others': 1}
        self.transform =  transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        self.device = torch.device("cpu")
        self.load_model()

    #load trọng số model và mô hình
    def load_model(self):
        self.model = resnet_18(class_names=self.class_names)
        self.model.load_state_dict(torch.load('model.pt', map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    # TODO: implement find label
    def find_label(self, img):
        img = Image.fromarray(img).convert('RGB')
        with torch.no_grad():
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            outputs = self.model(img)
            _, preds = torch.max(outputs, 1)
            label = preds.detach().cpu().numpy()[0]
        return self.labels[label]
