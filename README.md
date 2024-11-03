# CSE151A_Pokemon_Classification


## EDA:
- Bar Graph on the frequency of each class, winged and not winged done 
- Plot dimensions of images on a scatter plot
- Plot images after resizing all to an uniform size
- Case study on background diversity 
- Case study on different “types” of visual representations of the pokemon

## Data Preprocessing:
- Dropped bad data in dataset
  - Bad data refers to pokemon characters who are wingless species that have features that is similar to wings. We do so to avoid confusion for black box traininga to improve model performance.
- Make a csv file that maps pokemon name to (winged or not winged)
  - Since the original data is not labeled, we need to manually label the data to perform supervised classification.
- Resize the images to (224 x 224)
  - From EDA, we learned that the images vary in dimensions. Thus, we resized the images to ensure consistency in input size since we are considering to use Convolutional Neural Network (CNN).
- Grayscale the images
  - Since color is not an attribute that contribute to winged or not winged, we grayscale the images to reduce complexity and noises in the data.
- Normalize the pixel values
  - We might be using gradient descent to find optimal weight for features. If pixel values are in different ranges, it can lead to longer training times and difficulty to converge.
- Train/Val/Test splits/Cross-validation
  - This step sets up the data for model training.
- Data Augmentation 
- Flattening the Image (Logistic Regression)

Later steps:
## Model Training
### KNN
### CNN
- Experiment with architecture (ResNet-18, ResNet-54)
- Experiment with Optimizers (Adam, RMSProp, AMSGrad)
- Experiment with batch size (32, 64)
- Experiment with Epochs (25, 50)
