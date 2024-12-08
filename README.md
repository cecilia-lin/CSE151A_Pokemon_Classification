# CSE151A Pokemon Classification Report

## Introduction

In the modern era, where visual data dominates many aspects of technology, image classification has become a cornerstone of machine learning applications. This project focuses on the supervised classification of Pokémon characters as winged or wingless, using Kaggle's "7,000 Labeled Pokémon" dataset. This dataset, containing 150 Pokémon species with 30–45 images each, provides a fun and engaging platform to explore real-world challenges in image-based predictive modeling.

We chose this project because it combines technical rigor with creativity, making the learning process both practical and enjoyable. Building a model to classify visual features of Pokémon characters requires advanced techniques like Convolutional Neural Networks (CNNs) and image preprocessing (e.g., resizing, augmentation, normalization). Through experimentation with model architectures and optimizers, we’ll tackle challenges such as class imbalance and overfitting.

The broader impact of a strong predictive model extends far beyond Pokémon. Image classification powers technologies in fields such as healthcare, autonomous vehicles, and e-commerce, where precise visual understanding drives critical decisions. By applying machine learning techniques to this imaginative dataset, we demonstrate how predictive models can be both impactful and accessible, inspiring further innovation in computer vision.






### Figure 10: Example Predictions

#### Logistic Regression
![Logistic Regression Predictions](results/LogisticRegression/Pred_Examples.png)

*Figure 10.1: Example Pokémon predictions for Logistic Regression, showing true and predicted labels.*

#### Support Vector Machine (SVM)
![SVM Predictions](results/SVC/Pred_Examples.png)

*Figure 10.2: Example Pokémon predictions for SVM, showing true and predicted labels.*

#### Residual Network 18 (ResNet18)
![Resnet Predictions](results/Resnet18/Pred_Examples.png)

*Figure 10.4: Example Pokémon predictions for Residual Network 18, showing true and predicted labels.*



## Methods

### EDA

After deciding on our topic and dataset, we took some time to explore our dataset. At first glance, there seemed to be many instances of "bad" or "incorrectly formatted" data that needed to be manually checked and removed from our dataset. For example, many pokemon that are wingless have wing-like features that would work against training our model. Some images were also included in data formats that was unable to be read by PIL (Python Imaging Library). All of these types of images were manually checked and marked. Then we took a deeper look into the remaining, usable images. We looked into several types of parameters that would affect our model, such as winged/not winged frequency, size/dimensions, color, and background diversity.

- Winged/not winged: We first graphed the frequency of each of the two classes. Results showed that were much more examples of not winged than winged Pokemon. We would either have to sample an equal number of images from both sets or use a wegith balance optimizer to accomodate for this imbalance.

### Figure 1: Class Distribution
![Class Distribution](images/class_distribution.png)

*Figure 1: Class distribution showing imbalance between winged and not winged Pokémon.*

- Image Dimensions: Our images were of various dimensions, so these needed to be resized and standardized to be used with our model. A scatter plot of image dimensions was created before and after potential re-sizing to visualize the results.

- Color: Initially, we believed that grayscaling the images would simplify inputs for our models. However, we later learned that having colors in our images does matter. We looked into different methods of normalizing pixel data.

- Background/type diversity: When looking at different images of the same Pokemon, it is clear that different images come from different sources, for example 3D rendering, plushies, cartoon/anime, or pokemon cards. These differences, along with the associated image backgrounds of each type of picture, were analyzed. The images and their backgrounds needed to be normalized in order to eliminate the possibility of them affecting the results of our model.

### Data Preprocessing

Based on our EDA, we carried out data prepocessing to remove "bad data" from our dataset, normalized all of the remaining images, and manipulated our dataset to use with our models.

First, images that were manually pre-labeled as "bad data" were removed from our dataset. Some images in the dataset had formats that were unsupported by PIL that were also removed.

### Figure 2: Examples of 'Bad Data'
![Bad Data Examples](images/bad_data_examples.png)

*Figure 2: Examples of Pokémon images marked as 'bad data,' which were excluded during preprocessing.*

The remaining images were resized to all have the same pixel dimensions (224 x 224). Then, pixels in each of the images were normalized. Multiple functions were made to use different forms of pixel normalization for different models, such as z-score and min-max normalization. The results of each type of normalization were visualized by plotting 10 examples images for each method. Additionally, a function was created for data augmentation. Functions from the PIL used to implement flip/rotation/shifting changes were condensed into a single function to be used on certain images for training or testing.

### Figure 3: Image Dimensions Before and After Resizing
![Image Dimensions](images/image_dimensions.png)

*Figure 3: Scatter plots showing image dimensions before and after resizing to 224x224 pixels.*

### Figure 4: Resized Images
![Resized Images](images/resized_examples.png)

*Figure 4: Examples of Pokémon images after resizing to 224x224 pixels.*

### Figure 5: Normalized Images
![Normalized Images](images/normalized_images.png)

*Figure 5: Examples of Pokémon images after applying Z-score and Min-Max normalization.*

### Figure 6: Augmented Images
![Augmented Images](images/augmented_images.png)

*Figure 6: Examples of Pokémon images after applying data augmentation, including rotations, flips, and shifts.*

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


<!--#### Convolutional Neural Network (CNN)-->

#### Residual Network 18 (ResNet18)

ResNet18 is a neural network architecture that works well with image classification tasks. It's known for its ability to train deep networks and introduces skip connections to allow gradients to flow more effectively throughout its networks. It fits well for our image classification task. Two different optimizers, Adam and RMSProp, were extensively tested with different learning rates and weight decays to minimize loss and prevent overfitting.

- pretrained('True'): This option pretrains the model with weights trained on ImageNet, which allows the models to start with generalized visual features, only requiring fine-tuning for our project.

- lr(various): Various learning rates were tested to optimize training speed and overshooting to fit the best with our project.

- wd(various): Various weight decays were tested to reduce overfitting to fit the best with our project.

## Results

The performance of the classification models—Logistic Regression, Support Vector Machine (SVM), and Residual Network 18 (ResNet18)—was evaluated using precision, recall, f1-score, and accuracy on the training, validation, and test datasets. Below are the results:

### Logistic Regression
- **Training Accuracy**: 63%
- **Validation Accuracy**: 63%
- **Test Accuracy**: 58%
- **Observations**:
  - Logistic Regression achieved moderate precision and recall on the training and validation datasets.
  - On the test set, the model struggled with the minority class (winged Pokémon), showing poor precision (17%) and recall (53%).
 
### Figure 7: Confusion Matrices

#### Logistic Regression
![Logistic Regression Confusion Matrix](results/LogisticRegression/Confusion_Matrix.png)

*Figure 7.1: Confusion matrix for Logistic Regression showing classification performance on test data.*

### Support Vector Machine (SVM)
- **Training Accuracy**: 80%
- **Validation Accuracy**: 77%
- **Test Accuracy**: 69%
- **Observations**:
  - SVM demonstrated strong performance on the training and validation datasets, reflecting good generalization.
  - On the test set, its performance was better than Logistic Regression but still struggled with minority class precision (24%) and recall (62%).

#### Support Vector Machine (SVM)
![SVM Confusion Matrix](results/SVC/Confusion_Matrix.png)

*Figure 7.2: Confusion matrix for Support Vector Machine (SVM) showing classification performance on test data.*

### Residual Network 18 (ResNet18)
- **Training Accuracy**: 99%
- **Validation Accuracy**: 99%
- **Test Accuracy**: 98%
- **Observations**:
  - ResNet18 outperformed all other models, achieving nearly perfect accuracy across datasets.
  - It excelled at classifying the minority class, with a precision of 93% and recall of 94%, making it the best-suited model for this classification task.
 
#### Residual Network 18 (ResNet18)
![ResNet18 Confusion Matrix](results/ResNet18/Confusion_Matrix.png)

*Figure 7.3: Confusion matrix for Residual Network 18 (ResNet18) showing classification performance on test data.*

## Discussion

### EDA
When we first explored the type of files in the dataset, we realized that we will be using python's built in libraries like PIL, so we decided it would be beneficial to get rid of them as there were very few of them. 

Also, when we were looking at the images, we realized there were certain classes of water-type pokemons where there fins looked like wings, we decided to drop them as well as it could deviate the model from generalizing due to bad examples.

### Data Preprocessing
The images had different sizes, so we had to get them all to same size to use it as an input to a model. Also, the images were fairly huge in size, even as big as 1024x1024, and having bigger images mean more variables to learn, so, we decided to resize the images to 224x224. This number was chosen because we had an idea that we will try resnet, and that's the input size for resnet.

We also normalized pixel values for a swift training and did a Train/Validate/Test splits for 80/10/10 on the data.

Also, since the labels were "Winged" and "Not Winged", we made it a numeral value for model with 0 being "Not Winged" and 1 being "Winged"

### Models

1. **Logistic Regression**:
   We knew that using Logistic Regression would not be a good idea for this usecase because images often have non-linear relations and it also can't adjust for zoom in images, the idea behind doing this was to make sure the initial pre-processing works well and also get a good baseline model. As expected, the model did not perform well, so we had our eyes set on SVC to see how well of an improvement can we get.

#### Logistic Regression
![Logistic Regression Learning Curve](results/LogisticRegression/Accuracy_Size.png)

*Figure 8.1: Learning curve for Logistic Regression showing training and validation accuracy as a function of training set size.*

![Logistic Regression Hyperparameter Tuning](results/LogisticRegression/Error_Complexity.png)

*Figure 8.2: Error vs. regularization strength for Logistic Regression showing the optimal regularization parameter.*

![Logistic Regression F1-Score Tuning](results/LogisticRegression/F1_Score_Complexity.png)

*Figure 8.3: F1-score vs. regularization strength for Logistic Regression showing the best F1 performance.*

2. **SVM**:
   We again had an idea that SVM wouldn't be the best model to use since it again captures linear-relations, but it would be better that Logistic Regression since using kernel methods would allow us to have non-linear modifications for individual features. The idea was to test the new method we've learned in class and expreiment with how well it works. It did offer a significant improvement over the last method but still fell short on our expectations. So we continued on the next model we wanted to test, the Resnet18


#### Support Vector Machine (SVM)
![SVM Learning Curve](results/SVC/Accuracy_Size.png)

*Figure 9.1: Learning curve for Support Vector Machine (SVM) showing training and validation accuracy as a function of training set size.*

![SVM Hyperparameter Tuning](results/SVC/Error_Complexity.png)

*Figure 9.2: Error vs. regularization strength for Support Vector Machine (SVM) highlighting optimal parameters.*

![SVM F1-Score Tuning](results/SVC/F1_Score_Complexity.png)

*Figure 9.3: F1-score vs. regularization strength for SVM showing the best performance.*

4. **ResNet18**
   We knew that Deep Learning would outperform traditional machine learning models in this case as it excels at catching complex relations in images, we chose resnet specifically becuase of the amount of recommendations it had online, and we decided to start with the most basic 18-layer variant of it which delivered excellent results, so we chose it as our final model and didn't contiue to train complex models

#### Residual Network 18 (ResNet18)
![ResNet18 Learning Curve](results/Resnet18/Loss_Curve.png)

*Figure 10.1: Loss curve for Residual Network 18 showing training and validation loss as a function of epochs.*


### Key ML Learnings from this Project
1. **Class Imbalance**:
   - Logistic Regression struggled with classifying the minority class, as evidenced by their low precision and recall for winged Pokémon.
   - ResNet18 and SVM handled class imbalance better, with ResNet18 achieving the highest performance due to its ability to capture complex visual features.

2. **Overfitting**:
   - ResNet18 achieved perfect training accuracy, but its excellent validation and test performance indicate effective mitigation of overfitting through techniques like data augmentation and weight decay.

3. **Generalization**:
   - ResNet18 demonstrated the best generalization, with consistently high metrics across datasets.
   - SVM also generalized well but was less effective at handling the minority class compared to ResNet18.

4. **Minority Class Challenges**:
   - Except for ResNet18, all models faced significant challenges in accurately classifying the minority class (winged Pokémon), as reflected in low recall and precision values.

## Conclusion

ResNet18 was the best-performing model, achieving a test accuracy of 98% and excelling at classifying both majority and minority classes. This highlights the superiority of deep learning for image classification tasks with complex visual features. Future work could focus on fine-tuning ResNet18’s hyperparameters, exploring ensemble techniques, or expanding the dataset to include a broader range of Pokémon species for improved generalization.

## Statement of Collaboration

### Name: Ayush Singh

**Title**: Coder & Writer

**Contribution**:
   - I was a part of the inital discussions where we decided the topic, what we were going to do for EDA, Preprocessing and proposed to train the ResNet model

   - I designed the initial pipeline for Exploratory Data Analysis. It included getting rid of bad images, confusing labels, plotting the statistics, which helped us decide on the preprocessing pipeline.

   - I also designed the initial pipeline for data preprocessing, which included resizing the images, normalizing pixel values, gray-scaling (we removed it later), and preparing the data splits. Other members later built upon this to modify the pipeline.

   - I've trained and tuned the entire ResNet18 model, which included multiple experiments with optimizers, learning rates, and regularizations. This also resulted in our best performing and final model.

   - I've written the Discussion Section in this report along with Cecilia Lin

   - Apart from these, I've been been a part of our group discussions, and tackling any challenges that come along our way.

### Name: Cecilia Lin

**Title**: Co-Project Manager & Writer

**Contribution**:
   - I set up meetings and created agenda for the group. I wrote the first abstract of the project with Vincent, went to TA Esha's office hours to revise abstract.
     
   - I helped labeling the dataset and analyzed backgrounds of the images for Milestone 2.
     
   - I created learning curve and complexity graphs and wrote conclusions about our models for Milestone 3.
     
   - I wrote conclusions for our new models and discussions for Milestone 4.
     
   - I am actively involved with the project to make sure we meet all deadlines.

### Name: Cindy Hong

**Title**: Coder, Co-Project Manager, & Writer

**Contribution**:

- Participated in the initial discussions and proposed the idea of training Logistic Regression, SVM, and KNN models. 
- Contributed to the Exploratory Data Analysis (EDA) by analyzing data and incorporating the use of the os library to make the notebook executable on all team members' machines without modifying paths. 
- Enhanced the initial data preprocessing pipeline by wrapping preprocessing code into reusable functions for easier subsequent calls. Added Z-score and Min-max normalization methods to facilitate further experimentation. 
- Developed data augmentation functions to address data imbalance by oversampling through image rotation, translation, and flipping. 
- Created downsampling functions to address data imbalance (later discarded). 
- Wrapped the learning curve and complexity graph generation code into reusable functions for better accessibility. 
- Developed functions to visualize and print confusion matrices and to display prediction examples on training, validation, and test sets. 
- Trained and tuned the Logistic Regression and SVM models, experimenting with different regularization methods and strengths. 
- Attempted training and tuning the KNN model, using the ROC-AUC curve to determine the optimal value of k, but the model did not yield satisfactory results (later discarded). 
- Contributed to training and tuning of the ResNet18 model, experimenting with optimizers, but the performance was suboptimal (not adopted). 
- Trained and tuned a custom CNN model, but it did not perform well (later discarded). 
- Attended Cindy's office hours to seek project topic suggestions
- Visited Howard's office hours multiple times to discuss overfitting issues and experimented with possible solutions. 
- Contacted Professor Solares after class to request GPU resources for deep learning model training and sought advice regarding model sampling methods. 
- Actively participated in the project, frequently summarizing the upcoming milestone requirements, coordinating with team members for task distribution, and ensuring timely completion of all milestones. 
- Wrote the final version of the project abstract based on the initial draft. 


