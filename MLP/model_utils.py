from torch import nn

class SoftmaxClassifier(nn.Module):
    def __init__(self, d_in=28*28, d_out=10):
        super(SoftmaxClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(d_in, d_out)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
class MLP(nn.Module):
    def __init__(self, d_in=28*28, d_out=10):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, d_out)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
    
class ConvNet(nn.Module):
    def __init__(self, d_out=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.fc1 = nn.Linear(1024, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, d_out)
        
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_pool1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
class ConvNetSVHN(nn.Module):
    def __init__(self, d_out=10, c=3):
        super(ConvNetSVHN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(1600, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, d_out)
        
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_pool1(out)
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out