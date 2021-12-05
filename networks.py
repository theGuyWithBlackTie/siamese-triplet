import torch.nn as nn
import torch.nn.functional as F


#
# Declaring and defining the network 'EmbeddingNet' that will generate the latent representations of images.
# Latent representations are just fancy name for the embeddings that are generated by the network.
#
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # convlutional layer
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5), nn.PReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), nn.PReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # this layer generates the final embedding which is of just two dimensions thats why there is '2' in last linear layer
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 256), nn.PReLU(),
            nn.Linear(256, 256), nn.PReLU(),
            nn.Linear(256, 2)
        )


    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


#
# Declaring and defininf the Classification Net which would be served in establishing Baseline. This will classify the embeddings
# generated from Embedding Net into one of the ten classes.return
#
class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()

        # This is the EmbeddingNet class object
        self.embedding_net = embedding_net

        # Introducing non-linearity
        self.non_linear    = nn.PReLU()

        # Classification layer. Input size is 2 as thats the embedding size in Embedding Net.
        self.fc1           = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.non_linear(output)
        output = F.log_softmax(self.fc1(output), dim=-1)
        return output

    def get_embedding(self, x):
        return self.non_linear(self.embedding_net(x))



class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output_1 = self.embedding_net(x1)
        output_2 = self.embedding_net(x2)
        return output_1, output_2

    def get_embedding(self, x):
        return self.embedding_net(x)




class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()

        self.embedding_net = embedding_net

    
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)

        return output1, output2, output3


    def get_embedding(self, x):
        return self.embedding_net(x)