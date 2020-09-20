# Animal Recognition - Convolutional Neural Network (CNN)
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
- [Step 1 - Loading Data](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-1---loading-data)
- [Step 2 - Pre-processing](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-2---pre-processing)
- [Step 3 - Multi-layer Perceptron](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-3---multi-layer-perceptron)
- [Step 4 - Optimizer](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-4---optimizer)
- [Step 5 - Training the model](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-5---training-the-model)
- [Step 6 - Tensorboard](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-6---tensorbard)
- [Step 7 - Building CNN](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-7---building-cnn)
- [Step 8 - Optimization Techniques](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-8---optimization-tecnique)
- [Step 9 - Predict](https://github.com/densaiko/Recognize_Animals_CNN_Deep_Learning/blob/master/README.md#step-9---predictions)

### Step 1 - Loading Data
In this section, you will unzip the data from google drive and load the data into your notebook
```
from google_drive_downloader import GoogleDriveDownloader as gdd
```

### Step 2 - Pre-Processing
In this section, you have to set the pixel into 256 256, covert_to_tensor, reshape and normalize it.
- convert to tensor `tf.convert_to_tensort(data_test, np.float32)`
- reshape `tf.reshape(data_test_tf, [910, 256, 256, 3])`
- normalize `np.divide(data_test_tf, 255.0)`

### Step 3 - Multi-layer Perceptron
In this section, you have to build your own model version to predict the data. I have build 3 hidden layers such as
- First hidden layer, I use 512 perceptrons and relu activation
- Second hidden layer, I use 256 perceptrons and relu activation
- Third hidden layer, I use 128 perceptrons and relu activation

### Step 4 - Optimizer
In this section, you have to create your own optimizer.
Here I use `adam` for simple CNN model and `SGD` for the ResNet50 Model

### Step 5 - Training the model
In this section, you have to set the tensorboard and earlystopping in your model fit. Tensorboard is a dashboard of our result of training and earlystopping to stop the running model to overcome over fitting.

### Step 6 - Tensorbard
In this section, I will visualize the dashboard of my training to visualize the epoch_accuracy and epoch_loss. To load the tensorboard library
```
from tensorflow.keras.callbacks import TensorBoard
```

### Step 7 - Building CNN
In this section, I use ResNet50 model with some customization such as
- First hidden layer, I use 512 perceptrons, relu activation and kernel_regularizer using `l2`. I also use dropout function around 30% during the training
- Second hidden layer, I use 256 perceptrons, relu activation and kernel_regularizer using `l2`. I also use dropout function around 20% during the training

### Step 8 - Optimization Tecnique
In this section, I use SGD optimizer to optimize the model 

### Step 9 - Predictions
In this section, you will predict the test_data and copy the result into DPHI Deep Learning Bootcamp to check the accuracy

### Resources
- **Python Version:** 3.7.6
- **Tensorflow Version:** 2.3.0
- **Dataset:** [DPHI Bootcamp](https://drive.google.com/file/d/176E-pLhoxTgWsJ3MeoJQV_GXczIA6g8D/view)
