# SVHN Image Classification using Convolutional Neural Networks (CNN)

## Introduction

This project demonstrates a full machine learning pipeline designed to classify images from the Street View House Numbers (SVHN) dataset, which consists of over 600,000 digit images taken from real-world photos of house numbers, using Convolutional Neural Networks (CNNs). The pipeline spans from data acquisition to model evaluation and prediction on unseen data. CNNs are particularly suited for image classification tasks due to their ability to capture spatial hierarchies.

The codebase and approach were adapted and expanded from a Kaggle notebook: [SVHN Classification with CNN Keras (96% Accuracy)](https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc). Key steps were reproduced and extended to meet coursework requirements and enable model customization and improvement.

## Business Objectives

The business goal of this machine learning application is to develop a reliable and efficient digit recognition system that can be deployed in automated systems where accurate digit identification is critical. This could include:
- Automated reading of house numbers from images in mapping applications.
- Postal address digit recognition in sorting facilities.
- Input interpretation in smart meters and devices that display numeric information.

A high-performing model will not only classify digits accurately but also generalize well to unseen data. Our target is to exceed 90% classification accuracy on the test dataset and demonstrate the model's robustness to noise, blur, and real-world variance in digit appearance.

## ML Pipeline

### 1. Data Collection and Validation

We used the cropped MNIST-like 32-by-32 images centered around a single character dataset instead of the full number one due to reduce the pre-processing.
The dataset was downloaded onto a google drive then downloaded from the google drive to the notebook using `gdown`. The two datasets used were:
- `train_32x32.mat`: The primary dataset for training the CNN model.
- `test_32x32.mat`: A separate dataset reserved for evaluating final model performance.

```python
gdown.download("https://drive.google.com/uc?id=1FSwCLJzPzL4ZmihRgjIkh3DImRwM0UHb", "train_32x32.mat")
gdown.download("https://drive.google.com/uc?id=10E7i5m1HaGWZkte4P8WMouNoFj0dFHmv", "test_32x32.mat")
```

The data was loaded using `scipy.io.loadmat`, which reads MATLAB files into Python dictionaries. The images were initially stored in a 4D array with shape `(32, 32, 3, N)`, which was transposed to match the expected input shape for CNNs: `(N, 32, 32, 3)`.

Label processing included:
- Flattening the label array
- Re-mapping digit label '10' to '0', as per SVHN documentation
- Verifying data integrity through shape and dtype inspections

After normalization (scaling pixel values between 0 and 1), the data was split into training and validation subsets using an 80/20 split, ensuring the model's performance could be monitored during training.

### 2. Exploratory Data Analysis (EDA)

EDA focused on understanding the structure, balance, and quality of the dataset:
- Visual inspection of random samples confirmed clarity and label correctness
- Histogram plots revealed that the dataset is relatively balanced across all digit classes
- Pixel intensity distributions confirmed the images were appropriately scaled
- The label distribution was visualized using Seaborn count plots

EDA showed a well-distributed dataset across classes, with minor variations in sample count per class.

### 3. Model Building

The CNN architecture was inspired by the referenced Kaggle notebook and consists of:
- Three `Conv2D` layers with increasing filter depth with ReLU activation for local feature extraction
- MaxPooling after each convolution block to downsample feature maps and reduce overfitting
- Dropout layers to improve generalization and prevent overfitting
- A fully connected `Dense` layer 
- An output layer with `softmax` activation for multi-class classification

```python
aux_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                           activation='relu',
                           input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation='softmax')
])
```

### 4. Model Evaluation

The model was trained over several epochs with early stopping based on validation loss. Evaluation included:
- Plotting training vs. validation accuracy and loss over time
- Computing the final accuracy on the validation set (~91%)
- Visualizing a confusion matrix to identify misclassification trends

Hyperparameter tuning experiments included:
- Removed Flatten() and replaced it with GlobalAveragePooling2D() and changed the final dense layer to 32, to increase performance to allow for a larger sample size
- Varying the number of filters in convolutional layers (e.g., 32, 64, 128)
- Testing different dropout values (0.3, 0.4), we found the best values to be 0.3 and 0.4 as these had the highest accuracy and relatievly fast speed
- Adjusting the number of epochs and batch sizes
- Flipping the layers upside down (starting with 128 and going down to 32), this cause each epoch to take very long and resulted in a crash

Each trained model was logged, and the best-performing configuration was retained based on validation accuracy.

### 5. Prediction

The final model was used to predict labels on the test dataset (previously unseen data). For each prediction:
- The predicted label and true label were compared
- Class probabilities were visualized using bar plots
- Misclassified samples were examined to identify common causes of error

The model achieved:
- Test set accuracy: ~91.4%
- Strong performance on most digits, though some confusion persisted between similar-looking digits like 3 and 5

## Jupyter Notebook Structure

The notebook is structured as follows:
1. **Library Imports and Setup**: Import necessary libraries and configure global parameters
2. **Data Acquisition**: Download and load `.mat` files, prepare image arrays
3. **Preprocessing**: Label encoding, normalization, dataset splitting
4. **EDA**: Dataset inspection and visualization
5. **Model Definition**: Build CNN architecture using Keras
6. **Training and Validation**: Train the model, visualize learning curves
7. **Evaluation**: Evaluate performance using metrics and confusion matrix
8. **Prediction**: Make predictions on unseen data and visualize output

## Future Work

Given more time and resources, the following improvements could be pursued:
- **Data Augmentation**: Implement transformations (rotation, shift, zoom) to increase dataset variability
- **Transfer Learning**: Use pre-trained models like ResNet or MobileNet for enhanced performance
- **Model Optimization**: Implement Keras Tuner or grid search for hyperparameter optimization
- **Advanced Architectures**: Explore deeper CNNs or architectures like Capsule Networks for better generalization

## Libraries and Modules

- **TensorFlow/Keras**: Core framework for deep learning and CNN implementation
- **NumPy**: Numerical array operations for handling image matrices and labels
- **Matplotlib/Seaborn**: Visualization tools for EDA and performance metrics
- **SciPy.io**: Functionality for loading MATLAB `.mat` files
- **Scikit-learn**: Utility for splitting datasets, computing metrics, and preprocessing
- **gdown**: Downloads files directly from Google Drive for dataset access

## Challenges Faced

### Limited Resources on GitHub Codespaces

One of the primary challenges encountered was the limited memory capacity of GitHub Codespaces. During training, especially when attempting to increase model complexity or train for more epochs, the kernel frequently crashed. This constrained our ability to perform more extensive experiments and hyperparameter tuning.

## Unfixed Bugs

- Model misclassifies digits with high visual similarity (e.g., 3 vs 5, 7 vs 1) when resolution is low or images are partially obscured
- Slight class imbalance can cause skewed predictions in low-sample digits
- Training time is sensitive to batch size and may vary across environments

## Acknowledgements and References

- Dataset source: [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
- TensorFlow/Keras documentation: [https://keras.io](https://keras.io)
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- CNN architecture guidance: François Chollet's *Deep Learning with Python*
- AI Assistance: ChatGPT was used for editing markdown, structuring documentation, and code refinement support

## Conclusions

This project demonstrates the successful application of a Convolutional Neural Network for digit classification on the SVHN dataset. With a structured pipeline and iterative model refinement, we achieved high classification accuracy and produced a model that performs robustly on real-world image data.

The modular pipeline can be extended with more complex architectures or applied to similar OCR tasks. Through clear documentation, visualizations, and analysis, this project showcases the practical utility of CNNs in handling real-world computer vision problems.
