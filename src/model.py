import torch
import torch.nn as nn

# some config
LSTM_HIDDEN = 64 # input is 128 so we change the hidden layer to 64
LSTM_LAYER = 1 # we start with one layer only. this is one of hyper-parameter
batch_size = 4 # 4 ideal size for my laptop
learning_rate = 1e-3 # standard params, also to be treated as hyper-parameter
# num_classes = 12 # total number of classes
# Model

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self,num_classes):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = nn.LSTM(input_size=128, hidden_size=LSTM_HIDDEN)
        self.classifier = nn.Linear(LSTM_HIDDEN, num_classes)
        self.softmax = nn.Softmax(dim=1) # to get confidence score

    def forward(self, x):
        # TODO complete forward function
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        logits = self.softmax(logits)
        return logits