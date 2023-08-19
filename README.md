# Detecting-disease-on-maize-plant
Plant Disease Classification with TensorFlow
This project demonstrates a deep learning model built using TensorFlow and Keras for classifying plant diseases based on images. The model utilizes convolutional neural networks (CNNs) to achieve accurate classification. The code provided here includes data preprocessing, model creation, training, evaluation, and prediction functionalities.

Setup
To get started, you'll need to install the required dependencies. Make sure you have Python and TensorFlow installed.

bash
Copy code
pip install tensorflow
pip install keras-metrics
Dataset
The dataset used for this project contains images of various plant diseases. The dataset is organized into different disease categories. The images are preprocessed and loaded using the TensorFlow's ImageDataGenerator and image_dataset_from_directory utilities.

Model Architecture
The model architecture consists of multiple convolutional layers, followed by max-pooling layers and fully connected layers. The architecture is as follows:

Input Preprocessing

Resizing to a specified image size
Rescaling pixel values to the range [0, 1]
Data Augmentation

Random horizontal and vertical flips
Random rotations
Convolutional Layers

Multiple convolutional layers with ReLU activation
Max-pooling layers for down-sampling
Fully Connected Layers

Flattening the output from convolutional layers
Dense layers with ReLU activation
Output layer with softmax activation for classification
Training and Evaluation
The model is trained using the training dataset and validated using the validation dataset. The training progress is monitored, and training and validation accuracy and loss are visualized using matplotlib. Additionally, various evaluation metrics such as accuracy, precision, recall, true positive rate, false positive rate, and false negative rate are calculated during training.

Prediction
The trained model is used to make predictions on new images. Given an image, the model predicts the class of the plant disease it belongs to, along with a confidence score. Example predictions are visualized using matplotlib.

Usage
Make sure you have the required dependencies installed.
Prepare your dataset in the required directory structure.
Update the paths in the code to point to your dataset.
Adjust hyperparameters such as batch size, image size, and number of epochs if needed.
Run the code and observe the training progress and evaluation metrics.
Test the model's predictions using the provided visualization.
Saving the Model
The trained model can be saved for future use. The model is saved to the specified directory using the model version number.

Author
Bello Iteoluwakisi

License
This project is licensed under the MIT License.
