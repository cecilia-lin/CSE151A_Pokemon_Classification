# CSE151A_Pokemon_Classification


## Milestone2(EDA):[Milestone2.ipynb](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/blob/main/Milestone2.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/cecilia-lin/CSE151A_Pokemon_Classification">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Observe 'bad data' and 'incorrectly formatted data'
- Bar Graph on the frequency of each class, winged and not winged
- Plot example classes of winged and not winged images
- Figure out the meaning of image data's dimensions
- Plot dimensions of images on a scatter plot for comparisons
- Plot images after resizing to an uniform size
- Case study on background diversity 
- Case study on different “types” of visual representations of the pokemon

## Milestone3(Data Preprocessing):[Milestone3.ipynb](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/blob/main/Milestone3.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/cecilia-lin/CSE151A_Pokemon_Classification">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Data Cleaning: Drop bad data in dataset
  - Bad data refers to pokemon characters who are wingless species that have features that is similar to wings. We do so to avoid confusion for black box training to improve model performance.
- Labeling: Make a csv file that maps pokemon name to (winged or not winged)
  - Since the original data is not labeled, we need to manually label the data to perform supervised classification.
- Resize the images to (224 x 224)
  - From Milestone2, we learned that the images vary in dimensions. Thus, we should resize the images to ensure consistency in input size since we are considering to use Convolutional Neural Network (CNN).
- Grayscale the images **(Got feedback that colors do matter, so we converted them to RGB)**
  - Since color is not an attribute that contributes to winged or not winged, we can grayscale the images to reduce complexity and noise in the data.
- Normalize the pixel values
  - We might be using gradient descent to find optimal weight for features. If pixel values are in different ranges, it can lead to longer training times and difficulty to converge.
  - Also, normalization can reduce the effect of different backgrounds of the images on feature extraction.
  - We wrote separate functions for Zscore and Minmax to facilitate the subsequent application of different normalization methods to different models.
- Data balancing
  - sample equal number of images from both sets
  - or
  - use a weight balanced optimizer
- Train/Val/Test splits/Cross-validation
  - This step sets up the data for model training.
- Data Augmentation 
  - We use functions from the Image library to implement flip/rotate/shift changes to images and integrate these actions into a function.

### Conclusion & next step

To classify if a pokemon is winged or not, we first built a logistic regression model without tuning any parameters. The logistic regression model is highly overfitted to the training data that it has a lower accuracy for testing and validation data. When we compare it to a model that predicts randomly, it has lower accuracy than the model. Thus, we moved on to build a SVM model to see if it will be a more generalized model. SVM is more robust to outliers. The performance of SVM measured by validation accuracy increased 0.54 to 0.77. Something we can do to improve would be tuning the parameters of the model to find the sweet spot between complexity and accuracy. 

Next, we will be building a KNN model which is non-parametric. It does not have a training phase, so we can save on runtime and do not need to worry about hyperparameter tuning for the model. 

## Milestone4(Data Preprocessing):[Milestone4.ipynb](create link)
- Tune hyperparameters for machine learning models
- Train K-Nearest Neighbors (KNN) model as second model
  
### Conclusion & next step
The KNN model has a test accuracy of 0.66, and it does not perform as good as the SVM model, which has a test accuracy of 0.79. One way we could ultilize to improve our KNN model is to adjust the training size to prevent overfitting. We also revisted our work in building the first model of Logistic Regression and SVM to apply regularization. Regularization prevents our model from overfitting to the training dataset, allowing our model to generalize. The next model we are thinking of is deep learning models, such as CNN and ResNet-18, because the traditional ML models we built so far do not perform with an accuracy above 0.80. We would ultize deep learning methods for higher accuracy and more efficient feature extraction. 

### ML Models we trained on

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

### DL Models we are thinking of

#### CNN

- Experiment with Optimizers (Adam, RMSProp, AMSGrad)
- Experiment with batch size (32, 64)
- Experiment with Epochs (25, 50)


#### ResNet-18

- Experiment with architecture (ResNet-18, ResNet-54)
- Experiment with Optimizers (Adam, RMSProp, AMSGrad)
- Experiment with batch size (32, 64)
- Experiment with Epochs (25, 50)
