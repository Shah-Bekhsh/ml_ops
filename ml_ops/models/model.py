import torch
from torch import nn


class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """

    def __init__(self, in_features: int, out_features: int, hidden_units: int = 500, drop_p = 0.5) -> None:
        super().__init__()
        # Input layer -> hidden layer -> output layer
        self.hidden_units = hidden_units
        self.l1 = torch.nn.Linear(in_features, hidden_units)
        self.output = torch.nn.Linear(hidden_units, out_features)
        # Activation function
        self.r = torch.nn.ReLU()

        
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.l1(x)
        x = self.r(x)
        x = self.dropout(x)
        x = self.output(x)

        return nn.functional.log_softmax(x, dim=1)
    
def validation(model, testLoader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testLoader:
        images = images.resize_(images.size()[0], 784)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy

def training(model, trainLoader, testLoader, criterion, optimizer=None, epochs=10, print_every=40):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    steps = 0
    running_loss = 0

    for e in range(epochs):
        model.train()
        
        for images, labels in trainLoader:
            steps += 1
            
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, testLoader, criterion)
                
                print(
                    "Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testLoader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(testLoader))
                )
                
                running_loss = 0

                model.train()
    return model