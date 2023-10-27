
import torch
import speechbrain as sb

class Network(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilations, kernel_sizes, paddings, strides, adaptive, out_features, n_classes):
        super().__init__()
        self.threshold = 1.2
        clayers=[]
        self.n_classes = n_classes
        for i in range(len(in_channels)):
            clayers += [Conv(out_channels[i], in_channels[i],dilations[i],kernel_sizes[i],paddings[i],strides[i])]

        clayers += [Linear_Layer(in_features=out_channels[-1], out_features=out_channels[-1]*2)]
        clayers += [Classifier(in_features=out_channels[-1]*2, out_features=n_classes)]
        self.layers = torch.nn.ModuleList(clayers)
    def forward(self, x):

        for layer in self.layers:
              layer = layer.to(x.device)
              x = layer(x)
        return x

class Conv(torch.nn.Module):
    def __init__(self, out_channel, in_channel, dilation, kernel_size, padding, stride):
        super().__init__()

        self.layer = torch.nn.Conv2d(out_channels=out_channel, kernel_size=kernel_size, in_channels=in_channel, dilation=dilation, padding=padding, stride=stride)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d((2,2))
        self.batchnorm = torch.nn.BatchNorm2d(out_channel)
        self.threshold = 1.2
        self.num_epochs = 1
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)


    def forward(self, x):

        x = self.act(self.layer(x))
        out = self.pool(self.batchnorm(x))
        return out

class Linear_Layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.threshold =  1.2
        self.num_epochs = 1

    def forward(self, x):
        x= self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.layer(x)
        x = self.relu(x)
        norm = x.norm(2, 1, keepdim=True)
        x = x / (norm + 1e-8)

        return x


class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.act = torch.nn.LogSoftmax(dim=1)
        self.opt = torch.optim.Adam(self.parameters(),lr=0.001)
        self.threshold = 1.2
        self.num_epochs = 1

    def forward(self, x):

        out = self.act(self.layer(x))

        return out

 

