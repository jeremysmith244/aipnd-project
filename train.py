import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

''' Function that fits dense NN to flower photos provided by user'''


def validation(model, testloader, criterion):
    ''' Take a model, and compare predictions on testloaders data'''

    accuracy = 0
    test_loss = 0

    for images, labels in testloader:

        # Check for whether gpu to do forward propagation
        if inpts.gpu:
            device = torch.device('cuda')
            images, labels = images.to(device), labels.to(device)

        # Forward propagate test images
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Model's output is log-softmax, exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with
        # true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions,
        # just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()

    return test_loss, accuracy


def trainer(model, trainloader, testloader, criterion, optimizer, epochs=5,
            print_every=40):
    '''Train neural network on some data

    Arguments:

        model -- the pretrained nn network and classifier to be trained
        trainloader -- the data loader pointing to training data
        testloader -- the data loader pointing to validation data
        criterion -- the pytorch loss criterion
        optimizer -- the pytroch optimizer
        epochs -- default of five epochs, unless specified
        print_every -- default show train, test loss and accuracy every 40

    Outputs:

        model -- returns the trained model, on cpu even if trained on cuda
    '''

    steps = 0
    running_loss = 0
    for e in range(epochs):

        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:

            # Check for whether gpu to do forward propagation
            if inpts.gpu:
                device = torch.device('cuda')
                images, labels = images.to(device), labels.to(device)
                model = model.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader,
                                                     criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()

    if inpts.gpu:
        model = model.to(torch.device('cpu'))

    return model


''' Beginning of main function'''

# Creater parser object for collecting the inputs
parser = argparse.ArgumentParser(
    description='Required image directory, \
    optional: save_dir, arch, learning_rate, epochs, hidden_units',
)

# Collect the inputs needed to specify to model parameters with argparse
parser.add_argument('data_dir', action='store', help='Location of images for \
                    training. Note this expects you to have split the input\
                    directory into subdirectories labelled train, test and\
                    valid, and the further subdivided the files within those\
                    into 102 different directories, grouped by flower type.')

parser.add_argument('-save_dir', action="store", dest='save_dir', help='If this\
                    argument is passed, model state dict will be saved here')

parser.add_argument('-arch', action="store", dest='arch', help='Choose between\
                    the various vgg models. Options are vgg11, vgg11_bn, \
                    vgg16 and vgg16_bn and default is vgg11_bn')

parser.add_argument('-learning_rate', action="store", dest='lr', type=float,
                    help='Optional specify learning rate, default 0.001')

parser.add_argument('-epochs', action="store", dest='epochs', type=int,
                    help='Optional specify epochs, default is 2')

parser.add_argument('-hidden_units', action="store", dest='hid_un', type=int,
                    help='Optional specify hidden units, default is 4096')

parser.add_argument('-gpu', action="store", dest='gpu', help='Choose \
                    whether to use gpu for inference')

# Parse inputs to create object to call needed parameters
inpts = parser.parse_args()

# Create variables for the input directories
data_dir = inpts.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define transforms for the training, validation sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           [0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True)

validloader = torch.utils.data.DataLoader(
    valid_data, batch_size=32, shuffle=True)

# Check desired aritechture and load pretrained model from pytorch
if inpts.arch:

    if inpts.arch == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
        arch = inpts.arch
        if inpts.hid_un:
            hid_un = inpts.hid_un
        else:
            hid_un = 4096

    elif inpts.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        arch = inpts.arch
        if inpts.hid_un:
            hid_un = inpts.hid_un
        else:
            hid_un = 500

else:
    model = models.vgg11_bn(pretrained=True)
    arch = 'vgg11_bn'
    if inpts.hid_un:
        hid_un = inpts.hid_un
    else:
        hid_un = 4096

# Freeze parameters so don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Add a 102 output classifier, with option hidden units or default 4096
if inpts.arch:
    if inpts.arch == 'vgg11_bn':
        classifier = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(25088, hid_un)),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(hid_un, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
    else:
        classifier = nn.Sequential(
            OrderedDict([('fc1', nn.Linear(1024, hid_un)),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(hid_un, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
else:
    classifier = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(25088, hid_un)),
                     ('relu', nn.ReLU()),
                     ('fc2', nn.Linear(hid_un, 102)),
                     ('output', nn.LogSoftmax(dim=1))]))

# Make sure only to pass the newly added classifier parameters to the optimizer
model.classifier = classifier

# Create loss criterion
criterion = nn.NLLLoss()

# Create optimizer, with optional learning rate argument
if inpts.lr:
    optimizer = optim.Adam(model.classifier.parameters(), lr=inpts.lr)
    lr = inpts.lr
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    lr = 0.001

# Check epochs and run model
if inpts.epochs:
    model = trainer(model, trainloader, validloader, criterion, optimizer,
                    epochs=inpts.epochs)
else:
    model = trainer(model, trainloader, validloader, criterion, optimizer,
                    epochs=2)

# Check for save request, and save if requested
if inpts.save_dir:

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': [3, 224, 224],
                  'output_size': [102],
                  'hidden_units': hid_un,
                  'architecture': arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, inpts.save_dir + r'/checkpoint.pth')
