import pandas as pd
from pathlib import Path
import cv2
import numpy as np
data_dir = Path('./data')
train_data = pd.read_csv(data_dir/'training.csv')
#print(train_data)
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
#show_images(train_data, range(4))

##randomly show  some face with missing key points
#missing_any_data = train_data[train_data.isnull().any(axis=1)]
#idxs = np.random.choice(missing_any_data.index, 4)
#show_images(train_data, idxs)

train_df = train_data.dropna()
train_df.info()
