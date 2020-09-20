# Animal Recognition - Convutional Neural Network (CNN)
### This notebook is one of the assignment from Deep Learning Boot Camp by DPHI. I have done in this notebook such as:
- Loading the data and save it into appropriate variables
- Converting the array of data into tensor in four-dimensional array
- Normalizing the data from 0 to 1 to achieve consistency in dynamic range for a set of data, signals or images to avoid mental distraction and reduce the data redundancy
- Building the simple CNN model to predict the test_data using Adam as an optimizer
- Building the CNN model architecture customizing the ResNet50 model and get 99.94% 
- Using EarlyStopping to minimize the overfitting and check the interactive dashboard from Tensorboard to see the epoch_loss and epoch_accuracy

### Quickstart
This comand is used to install tensorflow
`pip install tensorflow`

### Task in the assignment
- [Step 1 - Loading Data] (##Step 1 - Loading Data)
- Step 2 - Pre-processing
- Step 3 - Multi-layer Perceptron
- Step 4 - Optimizer
- Step 5 - Training the model
- Step 6 - Tensorboard
- Step 7 - Building CNN
- Step 8 - Optimization Techniques
- Step 9 - Predict

## Step 1 - Loading Data
In this section, you will unzip the data from google drive and load the data into your notebook
```
from google_drive_downloader import GoogleDriveDownloader as gdd
```
