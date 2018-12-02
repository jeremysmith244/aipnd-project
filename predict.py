from PIL import Image
import numpy as np
import torch
import argparse
from collections import OrderedDict
from torch import nn
from torchvision import models
import json


def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Open the image and find size
    im = Image.open(image)
    width, height = im.size

    # Find the shortest axis and resize the image
    if height > width:
        im = im.resize((256, int(height/(width/256))))
    else:
        im = im.resize((int(width/(height/256)), 256))

    # Crop the image
    im = im.crop(((im.size[0] - 224) / 2,
                  (im.size[1] - 224) / 2,
                  im.size[0] - (im.size[0] - 224) / 2,
                  im.size[1] - (im.size[1] - 224) / 2))

    # Convert it to numpy float to do math on
    np_im = np.array(im).astype('float32')

    # Divide by the maximum, subtract the mean,
    # divide by the sd for each color channel

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    for i in range(3):
        np_im[..., i] = np_im[..., i]/np_im[..., i].max()
        np_im[..., i] -= means[i]
        np_im[..., i] = np_im[..., i]/stds[i]

    # Flip the matrix as suggested in instructions
    np_im = np_im.transpose((2, 0, 1))

    # Convert to pytroch tensor
    pt_im = torch.tensor(np_im)

    return pt_im


def predict(image, model, topk):
    ''' Predict the class of image using a trained deep learning model'''

    # Make prediction, needs a 4D input with just a 1
    output = model.forward(image.unsqueeze(0))
    ps = torch.exp(output)

    # Load the class dictionary and invert it
    class_dict = model.class_to_idx
    class_dict = dict([[v, k] for k, v in class_dict.items()])

    # Get the top 5, look up the indexes in the class dictionary
    prob, clas = ps.topk(topk)
    clas = [class_dict[x] for x in clas.tolist()[0]]
    if inpts.gpu:
        prob = prob.cpu()
    prob = prob.detach().numpy()[0]

    return prob, clas


def load_checkpoint(filepath):

    # Load model from checkpoint file based on arch type
    checkpoint = torch.load(filepath)
    if checkpoint['architecture'] == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif checkpoint['architecture'] == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['architecture'] == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif checkpoint['architecture'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['architecture'] == 'inception_v3':
        model = models.inception_v3(pretrained=True)

    # Load the classifier for the model
    classifier = nn.Sequential(
        OrderedDict([('fc1', nn.Linear(model.classifier.in_features,
                                       checkpoint['hidden_units'])),
                     ('relu', nn.ReLU()),
                     ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                     ('output', nn.LogSoftmax(dim=1))]))

    # Create optimizer
    model.classifier = classifier

    # Load state dictionary and class indices
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


parser = argparse.ArgumentParser(
    description='Required image directory, \
    optional: save_dir, arch, learning_rate, epochs, hidden_units',
    )

# Collect the inputs needed with argparse
parser.add_argument('img_loc', action='store', help='Location of image on\
                     which to run prediction.')

parser.add_argument('checkpoint', action='store', help='Location of model\
                     which to use for prediction.')

parser.add_argument('-top_k', action="store", dest='top_k', type=int,
                    help='If this argument is passed, will output top k \
                    classes, else will output top 5')

parser.add_argument('-category_names', action="store", dest='cat_name', help='Choose \
                    whether to output by flower name, or by index')

parser.add_argument('-gpu', action="store", dest='gpu', help='Choose \
                    whether to use gpu for inference')

inpts = parser.parse_args()

# Check for how many top inputs to output
if inpts.top_k:
    top_k = inpts.top_k
else:
    top_k = 5

# Load image file and the model file
im = process_image(inpts.img_loc)
model = load_checkpoint(inpts.checkpoint)

# Check for whether to run forward on gpu
if inpts.gpu:
    device = torch.device('cuda')
    im = im.to(device)
    model = model.to(device)

# Make prediction for the image
probs, classes = predict(im, model, top_k)

# Check for whether class labels are desired based on flowers
if inpts.cat_name:
    with open(inpts.cat_name, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[x] for x in classes]

# Print predictions
i = 0
for k in range(len(classes)):
    i += 1
    print('Number ' + str(i) + ' prediction is ' + str(classes[k]) +
          ' with a ' + str(probs[k]) + ' probability')
