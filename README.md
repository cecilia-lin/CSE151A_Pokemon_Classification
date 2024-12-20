![Banner](images/banner.png)

<div align="center">
   <em>Powering the Pokédex since 1996.</em>
</div>

<br>

# 🦜 CSE151A Pokémon Classification Report

[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&labelColor=150458)](https://pandas.pydata.org/pandas-docs/stable/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&labelColor=013243)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&labelColor=3394c8)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&labelColor=fcfcfc)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?logo=kaggle&labelColor=fcfcfc)](https://www.kaggle.com/)


## 📋 Table of Contents

- [Introduction](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-introduction)
- [Methods](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-methods)
   - [EDA Results](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#eda-results)
   - [Data Preprocessing](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#data-preprocessing)
   - [Models](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#models)
- [Results](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-results)
   - [Logistic Regression](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#logistic-regression)
   - [Support Vector Machine (SVM)](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#support-vector-machine-svm)
   - [Residual Network 18 (ResNet18)](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#residual-network-18-resnet-18)
- [Discussion](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-discussion)
- [Conclusion](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-conclusion)
- [Statement of Collaboration](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-statement-of-collaboration)
- [ChatGPT Usage](https://github.com/cecilia-lin/CSE151A_Pokemon_Classification/tree/main#-chatgpt-usage)


## 🌟 Introduction


In the modern era, where visual data dominates many aspects of technology, image classification has become a cornerstone of machine learning applications. This project focuses on the supervised classification of Pokémon characters as winged or wingless, using Kaggle's "7,000 Labeled Pokémon" dataset. This dataset, containing 150 Pokémon species with 30–45 images each, provides a fun and engaging platform to explore real-world challenges in image-based predictive modeling.


We chose this project because it combines technical rigor with creativity, making the learning process both practical and enjoyable. Building a model to classify visual features of Pokémon characters requires advanced techniques like Convolutional Neural Networks (CNNs) and image preprocessing (e.g., resizing, augmentation, normalization). Through experimentation with model architectures and optimizers, we’ll tackle challenges such as class imbalance and overfitting.


The broader impact of a strong predictive model extends far beyond Pokémon. Image classification powers technologies in fields such as healthcare, autonomous vehicles, and e-commerce, where precise visual understanding drives critical decisions. By applying machine learning techniques to this imaginative dataset, we demonstrate how predictive models can be both impactful and accessible, inspiring further innovation in computer vision.


## 🔨 Methods


### EDA Results


After deciding on our topic and dataset, we took some time to explore our dataset. At first glance, there seemed to be many instances of "bad" or "incorrectly formatted" data that needed to be manually checked and removed from our dataset. These were removed from the dataset. We also looked into several types of parameters that would affect our model, such as winged/non-winged frequency, size/dimensions, and background diversity.


- Winged/Non-Winged: We first graphed the frequency of each of the two classes. Results showed that there were many more examples of non-winged than winged Pokémon. We would either have to sample an equal number of images from both sets or use a weight balance optimizer to accommodate for this imbalance.


![Class Distribution](images/class_distribution.png)


*Figure 1: Class distribution showing imbalance between winged and non-winged Pokémon.*
<br />


- Image Dimensions: Our images were of various dimensions, so these needed to be resized and standardized to be used with our model.


![Image Dimensions](images/image_dimensions.png)


*Figure 2: Scatter plots showing image dimensions before and after resizing to 224x224 pixels.*
<br />


- Background/type diversity: When looking at different images of the same Pokémon, it is clear that different images come from different sources, for example, 3D rendering, plushies, cartoon/anime, or Pokémon cards. These differences, along with the associated image backgrounds of each type of picture, were analyzed. The images and their backgrounds need to be normalized in order to eliminate the possibility of them affecting the results of our model.


### Data Preprocessing


First, images that were manually pre-labeled as "bad data" were removed from our dataset. Some images in the dataset had formats that were unsupported by the Python Image Library (PIL) and were also removed. Examples of such "bad data" are included in Figure 5 below. These particular examples showcase non-winged Pokémon that have "wing-like" qualities. After some discussion, we decided to remove these types of images from the dataset.


![Bad Data Examples](images/bad_data_examples.png)


*Figure 3: Examples of Pokémon images marked as 'bad data,' which were excluded during preprocessing.*
<br />


Additionally, some images had different sizes, so we decided to resize all of the images to 224x224 pixels.


![Resized Images](images/resized_examples.png)


*Figure 4: Examples of Pokémon images after resizing to 224x224 pixels.*
<br />


Then, pixels in each of the images were normalized. Multiple functions were made to use different forms of pixel normalization for different models, such as z-score and min-max normalization. The results of each type of normalization were visualized by plotting 10 examples of images for each method.


![Normalized Images](images/normalized_images.png)


*Figure 5: Examples of Pokémon images after applying Z-score and Min-Max normalization.*
<br />


Additionally, a function was created for data augmentation. Functions from the `PIL` used to implement flip/rotation/shifting changes were condensed into a single function to be used on certain images for oversampling. Data augmentation for computer vision is a powerful set of techniques aimed at improving the model's robustness and generalizability. As our Pokémon dataset consists of many different backgrounds and contexts, data augmentation would not only introduce variations our CNNs can generalize across, but also more valuable data.


![Augmented Images](images/augmented_images.png)


*Figure 6: Examples of Pokémon images after applying data augmentation, including rotations, flips, and shifts.*
<br />


Finally, methods to manipulate the dataset for use with our models were implemented. To resolve the class imbalance between winged/non-winged (our two labels) Pokémon, we resorted to a simple random sampling. The dataset was also split into training/validation/testing sets with a 10% test split and an 80/20 training/validation split. To serialize our two classes, "winged" and "non-winged", so we gave each class a numerical classification, with 0 being "non-winged" and 1 being "winged". All of the data preprocessing and augmentation described above was then applied to the dataset for the appropriate models.


### Models


#### Logistic Regression


The first model we decided to use was Logistic Regression. Logistic Regression is a supervised learning algorithm used for binary classification. Since Logistic Regression takes 1-dimensional inputs, the image data was flattened into a 1D vector in order to fit into our model.


- `solver('lbfgs')`: This solving algorithm can handle large datasets and is the default sorting algorithm, which ended up being our choice as well.


- `penalty('l2')`: Our solving algorithm supports only L2 regularization or none. An L2 regulation was added to prevent overfitting and promote simpler models.


- `class_weight('balanced')`: Even though the input data for this model is randomly sampled and balanced during preprocessing, the 'balanced' option is chosen to automatically balance the weights of each class frequency.


- `C('best_C')`: From testing, the best regularization is evaluated, and the best regularization complexity is chosen and added as a parameter into the model.


#### Support Vector Machine (SVM)


SVM is a supervised learning algorithm used for both classification and regression, though we use it in this project for regression. SVM works by finding the optimal decision boundary (hyperplane) between two classes in a feature space. This type of classification is effective for classification tasks like ours with high-dimensional spaces and noisy data.


- kernel('rbf'): The RBF kernel is suitable for image data and can help distinguish subtle features in images, like shapes resembling wings.


- C('best_C'): As stated before, the best regularization complexity was chosen through testing.


- probability('True'): Probability estimates are enabled in order to provide a confidence score for each classification.


- class_weight('balanced'): Even though the input data for this model is randomly sampled and balanced during preprocessing, the 'balanced' option is chosen to automatically balance the weights of each class frequency.


#### Residual Network-18 (ResNet-18)


Convolutional Neural Networks (CNNs) — networks suited to learn kernels/filters on image data using the "convolution" layer — capture spatial information (notably from images) better than dense neural networks. Of this broad family of neural network architectures, Residual Networks (ResNets) — deep CNNs consisting of residual connections, allowing for better gradient flow — were popularized by their evidently strong performance in image tasks (2015). Because of their battle-tested performance, we opt to use ResNets, specifically ResNet-18 (18 convolutional layers) for our binary image classification task. Four different optimizers, Adam, AdamW, AMSGrad, and RMSprop, were extensively tested with different learning rates and weight decays to minimize loss and prevent overfitting.


- `pretrained('True')`: This option pre-trains the model with weights trained on ImageNet, which allows the models to start with generalized visual features, only requiring fine-tuning for our project.


- `lr(various)`: Various learning rates were tested to optimize training speed and overshooting to fit the best with our project.


- `wd(various)`: Various weight decays were tested to reduce overfitting to fit the best with our project.


## 📊 Results


We evaluated the performance of our selected classification models — Logistic Regression, Support Vector Machine (SVM), and Residual Network-18 (ResNet-18) — with precision, recall, f1-score, and accuracy on the training, validation, and test datasets. We highlight our results below.


### Logistic Regression
![Logistic Regression Results Report](results/LogisticRegression/Report.png)


*Figure 7.1: Results of Logistic Regression on both non-winged (0) and winged (1) Pokémon*
<br />


- **Training Accuracy**: 63%
- **Validation Accuracy**: 63%
- **Test Accuracy**: 58%
- **Key Observations**:
  - Logistic Regression achieved moderate precision and recall on the training and validation datasets.
  - On the test set, the model struggled with the minority class (winged Pokémon), showing poor precision (17%) and recall (53%).
 
![Logistic Regression Confusion Matrix](results/LogisticRegression/Confusion_Matrix.png)


*Figure 7.2: Confusion matrix for Logistic Regression showing classification performance on test data.*
<br />


![Logistic Regression Predictions](results/LogisticRegression/Pred_Examples.png)


*Figure 7.3: Example Pokémon predictions for Logistic Regression, showing true and predicted labels.*
<br />


### Support Vector Machine (SVM)
![SVM Results Report](results/SVC/Report.png)


*Figure 8.1: Results of Logistic Regression on both non-winged (0) and winged (1) Pokémon*
<br />


- **Training Accuracy**: 79%
- **Validation Accuracy**: 77%
- **Test Accuracy**: 65%
- **Key Observations**:
  - SVM demonstrated strong performance on the training and validation datasets, reflecting good generalization.
  - On the test set, its performance was better than Logistic Regression but still struggled with minority class precision (21%) and recall (61%).


![SVM Confusion Matrix](results/SVC/Confusion_Matrix.png)


*Figure 8.2: Confusion matrix for Support Vector Machine (SVM) showing classification performance on test data.*
<br />


![SVM Predictions](results/SVC/Pred_Examples.png)


*Figure 8.3: Example Pokémon predictions for SVM, showing true and predicted labels.*
<br />


### Residual Network 18 (ResNet18)
![ResNet18 Results Report](results/Resnet18/Report.png)


*Figure 9.1: Results of ResNet18 on both non-winged (0) and winged (1) Pokémon*
<br />


- **Training Accuracy**: 99%
- **Validation Accuracy**: 98%
- **Test Accuracy**: 98%
- **Key Observations**:
  - ResNet18 outperformed all other models, achieving nearly perfect accuracy across datasets.
  - It excelled at classifying the minority class (winged), with a precision of 95% and recall of 90%, making it the best-suited model for this classification task.
 
![ResNet18 Confusion Matrix](results/Resnet18/Confusion_Matrix.png)


*Figure 9.2: Confusion matrix for ResNet18 showing classification performance on test data.*
<br />


![Resnet Predictions](results/Resnet18/Pred_Examples.png)


*Figure 9.3: Example Pokémon predictions for Residual Network 18, showing true and predicted labels.*
<br />


## 💭 Discussion


### EDA


To begin building models, we first looked into our dataset. The data contained was mostly fine, but there were some instances of bad data that needed to be removed. To process images, we had to make use of  `PIL`. Some images in our dataset were included in data formats that was unable to be read by the `PIL`. We decided it would be beneficial to get rid of any unsupported file types as there were very few of them. All of these types of images were manually checked, marked, and removed from the dataset.

Additionally, many non-winged Pokémon have wing-like features that would work against training our model. For example, there were certain classes of water-type Pokémon with fins that resemble wings, so we decided to drop them from the dataset. Tricky data like this could cause the model to learn inefficiently, and we wanted to have a more clear distinction between the two classes. We manually removed any instances of data as well. Once the dataset was parsed of inaccurate data, we considered several attributes of the remaining images.


- Winged/Non-Winged: Because we found that we had unbalanced datsets, we would either have to sample an equal number of images from both sets or use a weight balance optimizer to accommodate for this imbalance. We decided to develop a function with data augmentation that oversample and split the train/test sets accordingly to deal with this issue. Some models that we used are also able to handle unbalanced classes. 


- Image Dimensions: Some images had different sizes, so we had to get them all to the same size to use them as an input to our models. Some images were fairly huge in size, even as big as 1024x1024, so we ultimately decided to resize the images.


- Color: Initially, we believed that grayscaling the images would simplify inputs for our models. However, we later learned that having colors in our images does matter after discussing our project with TAs during Office Hours. Instead, we looked into different methods of normalizing pixel data, such as min-max and z-score normalization.


- Background/type diversity: For each Pokémon in our dataset, we found that the images were sources from various differnt medias, for example 3D rendering, plushies, cartoon/anime, or Pokémon cards. Although the Pokémon may look the same regardless of what source they're imaged from, we found that their backgrounds can be much different. We theorized that this could affect our model and that we should remove the inconsistencies in the background in order to produce more accurate results. We decided to normalize the images and their backgrounds in order to eliminate the possibility of them affecting the results of our model.


### Data Preprocessing


First, images that were manually pre-labeled as "bad data" were removed from our dataset. Some images in the dataset had formats that were unsupported by PIL that were also removed. These particular examples showcase non-winged Pokémon that have "wing-like" qualities. After some discussion, we decided to remove these types of images from the dataset. These outliers would make it too hard for our models to accurately train on our dataset.


Regarding the images themselves, we decided to resize all images to 224x224. This number was chosen to reduce the size of most images and lessen the input data for our models. This decision was also made with future models like ResNet in mind, as an image size of 224x224 is the input size for ResNet.


Then, we normalized the images and their backgrounds. We made multiple functions to use different forms of pixel normalization for different models, such as z-score and min-max normalization. We visualized the results of each type of normalization and decided on the best method for each of our models. Discussing these results together allowed us to isolate the Pokémon from their backgrounds and simplify training for our models.


Finally, methods to manipulate the dataset for use with our models were implemented. We saw that there was an imbalance between the frequency of non-winged and winged Pokémon, so we wrote a function to handle data augmentation for models that couldn't directly handle unbalanced data, namely the LR and SVM models. For the training/validation/testing sets, we first split 10% as testing data. Since the dataset is unbalanced, we decided to oversample the remaining data through data augmentation and split the augmented data into 80/20 training/validation sets. This allowed us to get accurate results with our traditional machine learning models. ResNet, however, can handle unbalanced sets on its own, so no data augmentation was needed for the model. ResNet featured a more standard 80/10/10 train/test/validation split.


### Models


#### 1. Logistic Regression


Logistic Regression is not suitable for an image classification task, especially one with this much variation in training data (different Pokémon and different backgrounds). Logistic Regression fails to account for spatial relations in images, color, depth, and many more other attributes crucial for making accurate predictions. However, it would serve as a necessary baseline for our future models. We performed a hyperparameter sweep, focusing on regularization and prevent overfitting to obtain the best/most generalizable model, so we decided to test a range of training set sizes and regularization strength, such as the parameter `"C"`. The results are outlined in the following figures.


![Logistic Regression Learning Curve](results/LogisticRegression/Accuracy_Size.png)


*Figure 10.1: Learning curve for Logistic Regression*


![Logistic Regression Hyperparameter Tuning](results/LogisticRegression/Error_Complexity.png)


*Figure 10.2: Error vs. regularization strength for Logistic Regression showing the optimal regularization parameter.*


![Logistic Regression F1-Score Tuning](results/LogisticRegression/F1_Score_Complexity.png)


*Figure 10.3: F1-score vs. regularization strength for Logistic Regression*
<br />


As expected, the model did not perform well despite our best efforts. However, this was fine given our expectations. After developing our Logistic Regression model, we had our eyes set on SVM to see how good of an improvement we get.
<br />


#### 2. Support Vector Machine (SVM)


We again had an idea that SVM wouldn't be the best model to use since it again captures linear relations, but it would be better than Logistic Regression since using kernel methods would allow us to have non-linear modifications for individual features. The idea was to test the new method we've learned in class and experiment with how well it works with the methods we applied to our Logistic Regression model. Again, we followed the same parameter hypertuning process as before. The results are outlined below.


![SVM Learning Curve](results/SVC/Accuracy_Size.png)


*Figure 11.1: Learning curve for SVM*


![SVM Hyperparameter Tuning](results/SVC/Error_Complexity.png)


*Figure 11.2: Error vs. regularization strength for SVM*


![SVM F1-Score Tuning](results/SVC/F1_Score_Complexity.png)


*Figure 11.3: F1-score vs. regularization strength for SVM.*
<br />


In general, SVM offered a significant improvement over the last method in all metrics. However, the model still fell short of our expectations. After researching other traditional Machine Learning models, we looked into Deep Learning models instead. We agreed that Deep Learning would outperform traditional machine learning models in this case as it excels at catching complex relations in images. So we continued on to the next model we wanted to test, Resnet18.
<br />


#### 3. Residual Network 18 (ResNet18)


We chose ResNet specifically because of the number of recommendations it had online, and we decided to start with the most basic 18-layer variant. Working with ResNet required specific data preprocessing, which we integrated into our dataset. which delivered excellent results, so we chose it as our final model and didn't continue to train complex models.


![ResNet18 Learning Curve](results/Resnet18/Loss_Curve.png)


*Figure 12.1: Training and testing loss curve for ResNet18.*
<br />






### Key ML Takeaways


1. **Class Imbalance**:
   - Logistic Regression and SVM struggled with classifying the minority class, as evidenced by their low precision and recall for winged Pokémon.
   - ResNet18 handled class imbalance better, achieving the highest performance due to its ability to capture complex visual features.


2. **Overfitting**:
   - ResNet18 achieved perfect training accuracy, but its excellent validation and test performance indicate effective mitigation of overfitting through techniques like data augmentation and weight decay.
   - Despite our best efforts, including attempts with data augmentation, regularization, using simpler kernels, and various data preprocessing methods, the gap in accuracy between the training and test sets of SVM remained above 10%. However, the process of tuning the kernel allowed us to gain a better understanding of identifying overfitting through learning curves and loss curves.


3. **Generalization**:
   - ResNet18 demonstrated the best generalization, with consistently high metrics across datasets.
   - SVM also generalized well but was less effective at handling the minority class compared to ResNet18.


4. **Minority Class Challenges**:
   - Except for ResNet18, all models faced significant challenges in accurately classifying the minority class (winged Pokémon), as reflected in low recall and precision values.


## 👋 Conclusion


ResNet18 was the best-performing model, achieving a test accuracy of 98% and excelling at classifying both majority and minority classes. This highlights the superiority of deep learning for image classification tasks with complex visual features. Future work could focus on tweaking ResNet18’s hyperparameters, exploring ensemble techniques, or expanding the dataset to include a broader range of Pokémon species for improved generalization. Future directions can also entail using more powerful vision models like Vision Transformers (ViTs) or variants of ViTs. 

Additionally, expanding the dataset to include more Pokémon species could address class imbalance and improve generalization. Incorporating strategies to better handle class imbalance, such as refined sampling methods or targeted augmentation, might also enhance the model's ability to classify the minority class accurately.

Another promising avenue for future work involves leveraging transfer learning from models pre-trained on larger and more diverse datasets, such as ImageNet. This approach could further enhance ResNet18’s performance by providing a richer set of learned features, especially for underrepresented classes in the Pokémon dataset. Additionally, experimenting with generative models like GANs (Generative Adversarial Networks) to synthesize new Pokémon images could help augment the dataset and mitigate class imbalance. Exploring semi-supervised or self-supervised learning approaches could also be beneficial, enabling the model to leverage unlabelled data more effectively. Finally, integrating advanced interpretability techniques could provide insights into the decision-making process of the model, fostering trust and identifying potential areas for improvement.

Despite its strong performance, the study has limitations that warrant further exploration. The dataset's class imbalance posed challenges, particularly for underrepresented Pokémon species, potentially limiting the model's ability to generalize effectively. Additionally, the reliance on a single architecture, ResNet18, while effective, leaves room for exploring alternative models or ensembles that might capture nuanced visual features more robustly. Lastly, the relatively narrow scope of the dataset limits the applicability of the findings to broader or more diverse image classification tasks. Addressing these limitations could significantly enhance the model's robustness and versatility.

In all, this project effectively demonstrated the application of machine learning techniques on a creative and complex classification task. Starting with simpler models and progressing to a deep learning approach like ResNet18 provided a thorough exploration of image classification challenges. The results showcase the potential of these methods to address real-world problems, offering a strong basis for future work in image-based predictive modeling.


## 🥧 Statement of Collaboration


### Name: Ayush Singh


**Title**: Coder & Writer


**Contribution**:
   - I was a part of the initial discussions where we decided the topic, what we were going to do for EDA, Preprocessing and proposed to train the ResNet18 model


   - I designed the initial pipeline for Exploratory Data Analysis. It included getting rid of bad images, confusing labels,and plotting the statistics, which helped us decide on the preprocessing pipeline.


   - I also designed the initial pipeline for data preprocessing, which included resizing the images, normalizing pixel values, gray-scaling (we removed it later), and preparing the data splits. Other members later built upon this to modify the pipeline.


   - I've trained and tuned the entire ResNet18 model, which included multiple experiments with optimizers, learning rates, and regularizations. This also resulted in our best-performing and final model.


   - I've written the Discussion Section in this report along with Cecilia Lin


   - Apart from these, I've been a part of our group discussions, and tackling any challenges that come along our way.


### Name: Cecilia Lin


**Title**: Co-Project Manager & Writer


**Contribution**:
   - I set up meetings and created an agenda for the group. I wrote the first abstract of the project with Vincent and went to TA Esha's office hours to revise the abstract.
     
   - I helped label the dataset and analyzed the backgrounds of the images for Milestone 2.
     
   - I created learning curve and complexity graphs and wrote conclusions about our models for Milestone 3.
     
   - I wrote conclusions for our new models and discussions for Milestone 4.

   - I wrote and revised the conclusions in the Jupyter notebook.
  
   - I cleaned the format of notebook and arranged figures for README for better readability.
     
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
- Developed functions to visualize ResNet18's loss curves and other plots
- Attended Cindy's office hours to seek project topic suggestions
- Visited Howard's office hours multiple times to discuss overfitting issues and experimented with possible solutions.
- Contacted Professor Solares after class to request GPU resources for deep learning model training and sought advice regarding model sampling methods.
- Actively participated in the project, frequently summarizing the upcoming milestone requirements, coordinating with team members for task distribution, and ensuring timely completion of all milestones.
- Wrote the final version of the project abstract based on the initial draft.


### Name: Andrew Lu


**Title**: Writer


**Contribution**:


- Wrote the Methods section in the final report.
- Organized the final version of the report and supplemented all sections with final results.
- Attended Office Hours with group members to review Final Milestone requirements.
- Attended group discussions and meetings.

### Name: Franziska Olivia Kuch


**Title**: Writer and Code Commenter


**Contribution**:


- Added detailed comments to the codebase, making it accessible and easier to understand for external readers and team members. This included explaining complex functions, preprocessing steps, and model evaluation processes.

- Report Writing: Wrote the Conclusion section of the report, summarizing key findings and providing insights into the project's outcomes and future direction.

- Report Figures Integration: Integrated figures into the markdown report.

- Report Results Section: Contributed significantly to the Results section of the report by adding the results and providing explanations for them.

- General Collaboration: Participated in team discussions and reviewed the report for quality and consistency.


### Name: Vincent Tu


**Title**: Writer & Code Commenter


**Contribution**:

- Code commenting and documentation: Added type hints, docstrings, explanations of code to the ML pipeline (preprocessing, augmentation, modeling, training, etc) improve code readability and ease-of-use. 
- Report Writing: Wrote the introduction and revised the final report, improving sentence/paragraph structures and readability. Added supplementary information in all steps of our ML pipeline. Added section in Conclusion for future work and limitations. Added table of contents and organized sections.


## 🤖 ChatGPT Usage


ChatGPT: https://docs.google.com/document/d/1uZLX1wxNKX3_IAx7YvsYM7n8N7J13PedCZ3WOxa7LGM/edit?usp=sharing 
