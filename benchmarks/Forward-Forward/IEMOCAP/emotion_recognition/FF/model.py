
import torch
import speechbrain as sb

class Network(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilations, kernel_sizes, paddings, strides, adaptive, out_features, n_classes):
        super().__init__()
        self.threshold = 1.2
        self.layers=[]
        self.n_classes = n_classes
        for i in range(len(in_channels)):
            self.layers += [Conv(out_channels[i], in_channels[i],dilations[i],kernel_sizes[i],paddings[i],strides[i])]

        self.layers += [Linear_Layer(in_features=out_channels[-1], out_features=out_channels[-1]*2)]
        self.layers += [Classifier(in_features=out_channels[-1]*2, out_features=n_classes)]

    def forward(self, x):

        for layer in self.layers:
              x = layer(x)
        return x

    
    def predict(self, x):
        goodness = []
        for layer in self.layers:
            x = layer(x)
            # if "CNN" in  layer._get_name():
            #     # goodness += [torch.mean(x.pow(2),(1,2,3))]
            #     pass
            if  isinstance(layer.layer, torch.nn.Linear):
                goodness += [x.pow(2).mean(1).detach().cpu()]
        return sum(goodness).unsqueeze(1)
    # def prediction(self, x):

    #     goodness_per_label = []
    #     with torch.no_grad():
    #       for label in range(self.n_classes):
    #           h = overlay(x, label, self.n_classes)
    #           h = h.unsqueeze(1)
    #           goodness = []
    #           for i, layer in enumerate(self.layers):
    #                   if not isinstance(layer.layer, torch.nn.Linear):
    #                       h = layer(h)
    #                   else:
    #                       h = layer(h)
    #                       goodness_val = (h.pow(2).mean(1)).detach().cpu()
    #                       goodness += [goodness_val]
    #           goodness_per_label += [sum(goodness).unsqueeze(1)]
    #       goodness_per_label = torch.cat(goodness_per_label, 1)

        # return goodness_per_label.argmax(1)
    def train_layers(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            layer = layer.to(x_pos.device)
            h_pos, h_neg, _ = layer.train_layer(h_pos, h_neg)
        return h_pos, h_neg


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

    def train_layer(self, xpos, xneg):

        for j in range(self.num_epochs):
            gpos = torch.mean(self.forward(xpos).pow(2), dim=(1, 2, 3))
            gneg = torch.mean(self.forward(xneg).pow(2), dim=(1, 2, 3))
            loss = torch.log(1 + torch.exp(torch.cat([-(gpos - self.threshold), gneg - self.threshold]))).mean()

            self.opt.zero_grad()       #for calling optimizer from the layer itself
            loss.backward()
            self.opt.step()            #for calling optimizer from the layer itself

        hpos, hneg = self.forward(xpos).detach(), self.forward(xneg).detach()

        return hpos, hneg, loss

class Linear_Layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        self.threshold = 1.2
        self.num_epochs = 1

    def forward(self, x):
        x= self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.layer(x)
        x = self.relu(x)
        norm = x.norm(2, 1, keepdim=True)
        x = x / (norm + 1e-8)

        return x

    def train_layer(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), loss


class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.layer = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.act = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(),lr=0.001)
        self.threshold = 1.2
        self.num_epochs = 1

    def forward(self, x):

        out = self.act(self.layer(x))

        return out

    def train_layer(self, xpos, xneg):

        for j in range(self.num_epochs):
            gpos = self.forward(xpos).pow(2).mean(1)
            gneg = self.forward(xneg).pow(2).mean(1)
            loss = torch.log(1 + torch.exp(torch.cat([-(gpos - self.threshold), gneg - self.threshold]))).mean()

            self.opt.zero_grad()       #for calling optimizer from the layer itself
            loss.backward()
            self.opt.step()            #for calling optimizer from the layer itself

        hpos, hneg = self.forward(xpos).detach(), self.forward(xneg).detach()

        return hpos, hneg, loss


# def overlay(x,y,n_classes):

#     x_over = x.clone()
#     x_over[:, 0:n_classes, 0] = x.min()
#     zero_vec = torch.zeros(x.size(0)).long()
#     x_over[range(x.shape[0]), y, zero_vec] = x.max()

#     return x_over
