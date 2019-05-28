import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from data import FaceKeypointsDataset,Normalize,ToTensor,RandomHorizontalFlip,prepare_train_valid_loaders
from net import MLP, CNN

IMG_SIZE = 96
def show_keypoints(image, keypoints):
    r = 3
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(keypoints)):
        #print(keypoints[i,:])
        if np.isnan(keypoints[i,0])  or np.isnan(keypoints[i,1]):
            continue
        cv2.circle(image, (int(keypoints[i,0]), int(keypoints[i, 1])), r , (0,0,255),-1)
    cv2.imshow("keypoints", image)
    cv2.waitKey()

def show_images(df, indxs, ncols=5, with_keypoints=True):
    print(indxs)
    for i, idx in enumerate(indxs):
        print(idx)
        image = np.fromstring(df.loc[idx, 'Image'], sep=' ').astype(np.uint8)\
                .reshape(-1, IMG_SIZE) 
        #print(image)
        if with_keypoints:
            keypoints = df.loc[idx].drop('Image').values.astype(np.float32)\
                        .reshape(-1, 2)
        else:
            keypoints = []
        show_keypoints(image, keypoints)

def train(train_loader, valid_loader, model, criterion, optimizer,n_epochs=50, saved_model='model.pt'):
    '''
    Train the model
    
    Args:
        train_loader (DataLoader): DataLoader for train Dataset
        valid_loader (DataLoader): DataLoader for valid Dataset
        model (nn.Module): model to be trained on
        criterion (torch.nn): loss funtion
        optimizer (torch.optim): optimization algorithms
        n_epochs (int): number of epochs to train the model
        saved_model (str): file path for saving model
    
    Return:
        tuple of train_losses, valid_losses
    '''

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity

    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device))
            # calculate the loss
            loss = criterion(output, batch['keypoints'].to(device))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*batch['image'].size(0)

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for batch in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device))
            # calculate the loss
            loss = criterion(output, batch['keypoints'].to(device))
            # update running validation loss 
            valid_loss += loss.item()*batch['image'].size(0)

        # print training/validation statistics 
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss/len(train_loader.sampler.indices))
        valid_loss = np.sqrt(valid_loss/len(valid_loader.sampler.indices))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch+1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss 
    return train_losses, valid_losses

def predict(data_loader, model):
    '''
    Predict keypoints
    Args:
        data_loader (DataLoader): DataLoader for Dataset
        model (nn.Module): trained model for prediction.
    Return:
        predictions (array-like): keypoints in float (no. of images x keypoints).
    '''
    
    model.eval() # prep model for evaluation

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device)).cpu().numpy()
            if i == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))
    
    return predictions

def view_pred_df(columns, test_df, predictions, image_ids=range(1,6)):
    '''
    Display predicted keypoints
    Args:
        columns (array-like): column names
        test_df (DataFrame): dataframe with ImageId and Image columns
        predictions (array-like): keypoints in float (no. of images x keypoints)
        image_id (array-like): list or range of ImageIds begin at 1
    '''
    pred_df = pd.DataFrame(predictions, columns=columns)
    pred_df = pd.concat([pred_df, test_df], axis=1)
    pred_df = pred_df.set_index('ImageId')
    show_images(pred_df, image_ids)  # ImageId as index begin at 1 

data_dir = Path('./data')
train_data = pd.read_csv(data_dir/'training.csv')
test_data = pd.read_csv(data_dir/'test.csv')
datasets = {'L': ['left_eye_center_x', 'left_eye_center_y',
                  'right_eye_center_x','right_eye_center_y',
                  'nose_tip_x', 'nose_tip_y',
                  'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y',
                  'Image'
                 ], 
            'S': ['left_eye_inner_corner_x','left_eye_inner_corner_y', 
                  'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 
                  'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 
                  'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 
                  'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 
                  'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 
                  'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 
                  'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
                  'mouth_left_corner_x', 'mouth_left_corner_y', 
                  'mouth_right_corner_x', 'mouth_right_corner_y', 
                  'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
                  'Image'
                 ]
           }
#print(train_data)
# drop any-missing-keypoint records
train_df = train_data[datasets["L"]].dropna()
test_df = test_data
# how many samples per batch to load
batch_size = 128
# percentage of training set to use as validation
valid_size = 0.2

# Define a transform to normalize the data
tsfm = transforms.Compose([RandomHorizontalFlip(p=0.5, dataset="L"),Normalize(), ToTensor()])

# Load the training data and test data
trainset = FaceKeypointsDataset(train_df, transform=tsfm)
testset = FaceKeypointsDataset(test_df, train=False, transform=tsfm)

# prepare data loaders
train_loader, valid_loader = prepare_train_valid_loaders(trainset,valid_size,batch_size)
#model = MLP(input_size=IMG_SIZE*IMG_SIZE, output_size=30,
#        hidden_layers=[128,64], drop_p=0.1)
outputs = len(datasets["L"]) - 1
model = CNN(outputs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

'''
train_losses, valid_losses = train(train_loader, valid_loader, model,
                                   criterion, optimizer, n_epochs=50, 
                                   saved_model='model.pt')
'''
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
#Load the minimum valuation loss model
model.load_state_dict(torch.load('model.pt'))
predictions = predict(test_loader, model)
columns = train_df.drop('Image', axis=1).columns
view_pred_df(columns, test_df, predictions)

