# CSE151A Pokemon Classification Report

## Introduction

In the modern era, where visual data dominates many aspects of technology, image classification has become a cornerstone of machine learning applications. This project focuses on the supervised classification of Pokémon characters as winged or wingless, using Kaggle's "7,000 Labeled Pokémon" dataset. This dataset, containing 150 Pokémon species with 30–45 images each, provides a fun and engaging platform to explore real-world challenges in image-based predictive modeling.

We chose this project because it combines technical rigor with creativity, making the learning process both practical and enjoyable. Building a model to classify visual features of Pokémon characters requires advanced techniques like Convolutional Neural Networks (CNNs) and image preprocessing (e.g., resizing, augmentation, normalization). Through experimentation with model architectures and optimizers, we’ll tackle challenges such as class imbalance and overfitting.

The broader impact of a strong predictive model extends far beyond Pokémon. Image classification powers technologies in fields such as healthcare, autonomous vehicles, and e-commerce, where precise visual understanding drives critical decisions. By applying machine learning techniques to this imaginative dataset, we demonstrate how predictive models can be both impactful and accessible, inspiring further innovation in computer vision.

## Figures

## Methods

### EDA

After deciding on our topic and dataset, we took some time to explore our dataset. At first glance, there seemed to be many instances of "bad" or "incorrectly formatted" data that needed to be manually checked and removed from our dataset. For example, many pokemon that are wingless have wing-like features that would work against training our model. Some images were also included in data formats that was unable to be read by PIL (Python Imaging Library). All of these types of images were manually checked and marked. Then we took a deeper look into the remaining, usable images. We looked into several types of parameters that would affect our model, such as winged/not winged frequency, size/dimensions, color, and background diversity.

- Winged/not winged: We first graphed the frequency of each of the two classes. Results showed that were much more examples of not winged than winged Pokemon. We would either have to sample an equal number of images from both sets or use a wegith balance optimizer to accomodate for this imbalance.

- Image Dimensions: Our images were of various dimensions, so these needed to be resized and standardized to be used with our model. A scatter plot of image dimensions was created before and after potential re-sizing to visualize the results.

- Color: Initially, we believed that grayscaling the images would simplify inputs for our models. However, we later learned that having colors in our images does matter. We looked into different methods of normalizing pixel data.

- Background//type diversity: When looking at different images of the same Pokemon, it is clear that different images come from different sources, for example 3D rendering, plushies, cartoon/anime, or pokemon cards. These differences, along with the associated image backgrounds of each type of picture, were analyzed. The images and their backgrounds needed to be normalized in order to eliminate the possibility of them affecting the results of our model.

### Data Preprocessing

Based on our EDA, we carried out data prepocessing to remove "bad data" from our dataset, normalized all of the remaining images, and manipulated our dataset to use with our models.

First, images that were manually pre-labeled as "bad data" were removed from our dataset. Some images in the dataset had formats that were unsupported by PIL that were also removed.

The remaining images were resized to all have the same pixel dimensions (224 x 224). Then, pixels in each of the images were normalized. Multiple functions were made to use different forms of pixel normalization for different models, such as z-score and min-max normalization. The results of each type of normalization were visualized by plotting 10 examples images for each method. Additionally, a function was created for data augmentation. Functions from the PIL used to implement flip/rotation/shifting changes were condensed into a single function to be used on certain images for training or testing.

Finally, methods to manipulate the dataset for use with our models were implemented. Random sampling from both winged and non winged pokemon was decided to be used as the method to resolve the imbalances between the two classes of pokemon. The dataset was also split into training/validation/testing sets with a 80/10/10 split, respectively. All of the data preprocessing and augmentation described above was then applied on the appropriate sets.

### Models

#### Logistic Regression

The first model we decided to use was Logistic Regression. Logistic Regression is a supervised learning algorithm used for binary classification. Since Logistic Regression takes 1-dimensional inputs, the image data was flattened into a 1D vector in order to fit into our model.

- max_iter('100'): This defines the maximum number of iterations taken for the solvers to converge. We chose a default value of 100.

- solver('lbfgs'): This solving algorithm can handle large datasets and is the default sorting algorithm, which ended up being our choice as well. 

- penalty('l2'): Our solving algorithm supports only L2 regularization or none. An L2 regulation was added to prevent overfitting and promote simpler models.


- class_weight('balanced'): Even thought the input data for this model is randomly sampled and balanced during preprocessing, the 'balanced' option is chosen to automatically balance the weights of each class frequency.

- C('best_C'): From testing, the best regularization is evaluated, and the best regularization complexity is chosen and added as a parameter into the model. 

#### SVM (Support Vector Machine)

SVM is a supervised learning algorithm used for both classification and regression, though we use it in this project for regression. SVM works by finding the optimal decision boundary (hyperplane) between two classes in a feature space. This type of classification is effective for classification tasks like ours with high-dimensional spaces and noisy data.

- kernel('rbf'): The RBF kernel is suitable for image data and can help distinguish subtle features in images, like shapes resembling wings.

- C('best_C'): As stated before, the best regularization complexity was chosen through testing.

- probability('True'): Probability estimates are enabled in order to provide a confidence score for each classification.

- class_weight('balanced'): Even thought the input data for this model is randomly sampled and balanced during preprocessing, the 'balanced' option is chosen to automatically balance the weights of each class frequency.

#### K-Nearest Neighbors (KNN)

K-Nearest Neighbors is an instance-based learning algorithm that sorts data points into classes based on the classes of its nearest neighbors in the feature space. KNN is able to classify a Pokemon as winged or not-winged based on its image features.

- n_neighbors('k_best'): Based on testing, the best performing k value under the ROC AUC will be used in the model.

- p('1'): A p-value of 1 is chosen to indicate the use of Manhattan Distance, which works well with high-dimensional data like ours.

<!--#### Naive Bayes Regression-->

<!--#### Convolutional Neural Network (CNN)-->

#### Residual Network 18 (ResNet18)

ResNet18 is a neural network architecture that works well with image classification tasks. It's known for its ability to train deep networks and introduces skip connections to allow gradients to flow more effectively throughout its networks. It fits well for our image classification task. Two different optimizers, Adam and RMSProp, were extensively tested with different learning rates and weight decays to minimize loss and prevent overfitting.

- pretrained('True'): This option pretrains the model with weights trained on ImageNet, which allows the models to start with generalized visual features, only requiring fine-tuning for our project.

- lr(various): Various learning rates were tested to optimize training speed and overshooting to fit the best with our project.

- wd(various): Various weight decays were tested to reduce overfitting to fit the best with our project.

## Results

## Discussion

## Conclusion