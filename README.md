# DL_COVID-19_prediction_using_Chest_X_Ray

The role of chest radiography in predicting COVID-19

Introduction

The COVID-19 pandemic continues to have a devastating effect on the health and well-being of the global population. A critical step in the fight against COVID-19 is the effective screening of infected patients, with one of the key screening approaches being radiology examination using chest X- ray images. It was found in early studies that patients that show abnormalities in chest radiography images are characteristic of those infected with COVID-19[21]. Motivated by this and inspired by the work done by researchers so far, in this study I have designed a deep convolutional neural network (CNN) model along with five other pre-trained models namely, MobileNet, MobileNetV2, VGG-16, VGG-19, Inception V3 for detecting COVID-19 cases from chest X-ray (CXR) images. The objective behind designing these models is the development of highly accurate yet practical deep learning solution that can be a helpful diagnostic screening tool for radiologists in the early diagnosis of the disease thereby ensuring that people are not tested unnecessarily using invasive tests, unnecessarily loaded with medication or isolated, while making sure that they are able to get the right treatment quickly which is indispensable to prevent the spread of SARS-COV-2. 

Methodology

Dataset: The public dataset shared on the GitHub website by Joseph Paul Cohen [19] was utilized as the dataset for this image classification problem. Also, a split ratio of 75%,15%, and 10% respectively for the training, validation, and test datasets was considered. The CXR images consisted of two classes, normal (1650 images) and COVID-19 (920 images). The images belonging to both these classes were equally split between test, training, and validation datasets. 

Data Preprocessing: The data preprocessing steps discussed below were kept the same for all six deep learning models. Keras Image Data Generator's inbuilt image augmentation functionality was leveraged with the following augmentation parameters: rotation range (+/-0.2), vertical flip, zoom range of 0.2, shear range of 0.2. Data Augmentation was performed only for training and validation dataset. However, rescaling was performed for all the images in test , training, and validation set by dividing the image sizes by 255 [39]. Keras Image Data Generator's inbuilt image method, flow_from_directory() was utilized for setting the input image size of 255 x 255, set the batch size of 64, shuffling the images, as well as set the class mode of ‘binary’.

Models deployed: CNN and five other transfer learning models, MobilNet, MobilNetV2, VGG-16, VGG-19, Inception V3 were utilized to address this supervised image classification problem. The choice of these six models was based on research already conducted by data scientists for this particular image classification problem[25][39][32][35]. 

●	The CNN model deployed was a sequential model consisting of 3 convolutional layers, followed by 3 dropout layers, 3 max-pooling layers, and 2 dense layers at the end. 

●	All five transfer learning models deployed were initially built on the pre-trained on the ImageNet dataset followed by 4 dense layers and 3 dropout layers at the end. For the exact layout of all the models refer to the code posted on my Github profile.


Building the model For all the models the following hyperparameters and metrics were employed: 
-	‘Adam’ was used as an optimizer with a learning rate of 0.0001
-	‘Sigmoid’ was used as the activation function for the output layer. ReLU was employed as activation functions for the other layers to avoid forwarding of any negative values through the network[35].
-	Being a supervised classification problem, the system performance was evaluated mainly by using accuracy and recall[33][34]. Furthermore, precision and F1-score were also computed for each class (COVID-19, and Normal) to obtain a more realistic result as the dataset is slightly imbalanced.
-	Number of epochs of 500, batch size of 64, and patience of 5 were used for training the model.
-	Three callbacks, namely Early Stopping, Monitor, and Learning Rate Scheduler were defined for all the models to further prevent any issues with regard to overfitting.

Results
 
Both Inception V2 and MobileNet are good models for predicting whether a person has COVID-19 or not mainly because of the following reasons:

The pros are as follows: 

•	They yield a high recall and accuracy in the detection of COVID-19 cases. 

•	They have low loss and are able to generalize well to unseen datasets.

•	They both are lightweight models with few parameters that offer a fast screening tool with the ability to diagnose COVID-19. 

The con are as follows: 


•Model is trained with the limited number of COVID-19 cases because there are still not enough COVID-19 cases in datasets. As the datasets are updated with the increasing number of cases, more important features of COVID-19 can be learned by the model. 
Although both the models are performing well both on the validation and test datasets we propose MobileNet should be used for predicting COVID-19 in the future based on the CXR images mainly because it has the lowest runtime and it will be able to detect whether a person has COVID-19 or not with the high accuracy fast.

Conclusion

MobileNet model using the pre-trained ImageNet dataset is the best performing model to detect the disease from medical images.


Path Forward

In future studies, we plan to re-validate our model when the number of cases in public data sets is updated. Also, we plan to place our model in the cloud to assist doctors to promptly diagnose, isolate and treat COVID-19 patients. Thus, it will also be beneficial for health centers located in areas with low population, where there are not enough medical equipment and health personnel. 
Video link to the presentation: https://youtu.be/78VahgD51BU
