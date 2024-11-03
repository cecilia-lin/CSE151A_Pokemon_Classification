# CSE151A_Pokemon_Classification


## Milestone2(EDA):
- Observe 'bad data' and 'incorrectly formatted data'
- Bar Graph on the frequency of each class, winged and not winged
- Plot example classes of winged and not winged images
- Figure out the meaning of image data's dimensions
- Plot dimensions of images on a scatter plot for comparisons
- Plot images after resizing to an uniform size
- Case study on background diversity 
- Case study on different “types” of visual representations of the pokemon

## Milestone3(Data Preprocessing):
- Drop bad data in dataset
  - Bad data refers to pokemon characters who are wingless species that have features that is similar to wings. We do so to avoid confusion for black box training to improve model performance.
- Make a csv file that maps pokemon name to (winged or not winged)
  - Since the original data is not labeled, we need to manually label the data to perform supervised classification.
- Resize the images to (224 x 224)
  - From Milestone2, we learned that the images vary in dimensions. Thus, we should resize the images to ensure consistency in input size since we are considering to use Convolutional Neural Network (CNN).
- Grayscale the images
  - Since color is not an attribute that contributes to winged or not winged, we can grayscale the images to reduce complexity and noise in the data.
- Normalize the pixel values
  - We might be using gradient descent to find optimal weight for features. If pixel values are in different ranges, it can lead to longer training times and difficulty to converge.
  - Also, normalization can reduce the effect of different backgrounds of the images on feature extraction.
- Data balancing
  - sample equal number of images from both sets
  - or
  - use a weight balanced optimizer
- Train/Val/Test splits/Cross-validation
  - This step sets up the data for model training.
- Data Augmentation 






## Later steps:
## Model Training
### Traditional ML
#### KNN
#### SVM
#### Random Forest
#### Naive Bayes
### CNN
- Experiment with architecture (ResNet-18, ResNet-54)
- Experiment with Optimizers (Adam, RMSProp, AMSGrad)
- Experiment with batch size (32, 64)
- Experiment with Epochs (25, 50)
