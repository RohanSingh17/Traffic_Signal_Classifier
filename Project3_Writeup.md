## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[//]: # (Image References)

[image1]: ./Unique_Labels.jpg "Unique labels"
[image2]: ./Dataset_Histogram.jpg "Histogram of labels"
[image3]: ./grayscale_conv.jpg "Grayscale conversion example"
[image4]: ./hist_equ.jpg "Histogram equaliztion example"
[image5]: ./LENET.jpg "LeNet Architecture"
[image6]: ./Modified_CNN_Architecture.jpg "Modified CNN Architecture"
[image7]: ./GTS_downld.jpg "German traffic sign images"
[image8]: ./Predictions/70kph_pred.jpg "Prediction of 70kph sign"
[image9]: ./Predictions/Yield_pred.jpg "Prediction of Yield with background sign"
[image10]: ./Predictions/Yield_new_pred.jpg "Prediction of Yield with no background sign"
[image11]: ./Predictions/Walk_pred.jpg "Prection of pedestrian sign"
[image12]: ./Predictions/Turn_left_pred.jpg "Prediction of Turn left sign"
[image13]: ./Predictions/Priority_road_pred.jpg "Prediction of priority road sign"
[image14]: ./Predictions/No_entry_pred.jpg "Prediction of No-entry sign"
[image15]: ./Predictions/General_danger_pred.jpg "Prediction of General caution sign"
[image16]: ./Predictions/Priority_road_Convolution_Layer1.jpg "Priority road Convolution layer 1 feature map"
[image17]: ./Predictions/Priority_road_Convolution_Layer2.jpg "Priority road Convolution layer 2 feature map"
[image18]: ./Predictions/Pedestrian_Convolution_Layer1.jpg "Pedestrian sign Convolution layer 1 feature map"
[image19]: ./Predictions/Pedestrian_Convolution_Layer2.jpg "Pedestrian sign Convolution layer 2 feature map"
[image20]: ./Predictions/Test_PRF.jpg "Precision, recall and F-score on test dataset"



Overview
---
In this project we learned how to build a traffic sign recognition program. This program uses Deep learning methodology 
to recognise German traffic signs by itself. This methodology includes training a Convolutional neural network using dataset provided
for the project. The ipython notebook `Traffic_Sign_Classifier-Final` containing final code is discussed in 
this report. 

### DataSet Exploration
The dataset consisted of German traffic signs which were splitted in training, validation and testing sets as shown in the 
table below. Each of the sets consisted of the image in 32x32x3 format for network training. The number of samples provided 
in each of the datasets is as shown in the table below 

| Set       | No of Examples  | 
|:-------------:|:-------------:| 
| Test     | 34799      | 
| Validation      | 4410      |
| Test     | 12630     |

The German traffic sign dataset consisted of 43 unique labels. The snapshot of all the unique traffic signs
are shown in the image below.

![alt text][image1]

To know the frequencies of individual labels present in the all the datasets a histogram plot for the labels 
present in all the datasets is shown below. All the datasets have relatively lower number of samples per 
label for labels beyond class Id 20.

![alt text][image2]

## Design and Test a model architecture
The designing and testing a model architecture includes follwoing processes as described below:

#### 1. Preprocessing of the Images : 
includes following steps which helped improve the algorithm training accuracy and hence improved validation accuracy.
* **Grayscale** : This includes conversion of a colorscale image to grayscale
![alt text][image3]
* **Normalization** : This step is used to normalize all the pixels in the image within a certain specified
mean and variance. This ensures algorithm can easily converge as it does not have deal with pixels high range
of intensity variations in the image.

* **Histogram Equalization** : This is one of the steps which is used for contrast adjustment. It distributes
 energy evenly across all the bins of the image histogram. I have used an adaptive histogram equalization technique
 called Contrast limited adaptive histogram equalization (CLAHE) which is used to equalize contrast locally 
 across the whole image. An example of image contrast equalization is shown below.
 ![alt text][image4]
 
 #### 2. Model Architecture :
A convolutional neural network architecture which is commonly know as LeNet-5 architecture was used as the
starting point. The image of LeNet-5 architecture is shown below _`(LeNet. source - Yann Lecun)`_
 
 
 ![alt text][image5] 
 
The **maximum validation accuracy** achieved using the above architecture was around `93-94%` when used with some 
preprocessing techniques as described above. 

 ![alt text][image6]

Hence, in order to improve the accuracy of the network further some changes to the LeNet architecture was made.
The changes were made to the network such that fully connected prediction layers gets to visualize lower 
level features which were recognized by convolutuion layers. This is different from LeNet architecture where
fully connected layers gets to visualize refined features from the last convolution layer. Even the number 
layers at each of the convolutional layer are increased. Thus an alternate architecture was created which 
was able to achieve a maximum validation accuracy of around `98%` consistenly.

 #### 3. Model training :
The training of the model also included tuning hyper parameters. The values of hyper parameters for the tuned 
model are as shown in the table below. 

| Hyper Parameter    |       Value  | 
|:-------------:|:-------------:| 
| Learning Rate     | 0.001      | 
| Epoch      | 150      |
| Batch Size     | 128     |
| Keep probability (Dropout)  | 0.8     |

* **_Learning rate_** was one of the parameters which had maximum affect on how fast model converged. I tried range of values 
from 1e-4 to 1e-3. 5e-4 learning rate converged within 50 to 60 epochs but oscillated around 97.3% accuracy 
1e-4 was slow to converge but maximum accuracy achieved was around 96.9%. The maximum accuracy achieved was
using 1e-3.

* **_Epochs_** were taken to be 150 although model converged to 98% validation accuracy around epoch 52.

* **_Batch Size_** had little effect on the accuracy of prediction. However larger batch sizes resulted in 
slightly lower validation accuracies. 

* **_Dropout Rate_** was taken to be around 0.8 as it ensured model didn't achieved 100% training accuracy.
A lower value was not able to achieve higher accuracies.

 #### 4. Solution Approach : 
As described earlier in step 2 LeNet-5 CNN architecture was used as the starting point and maximum
validation accuracy of around 90% was achieved. To improve the accuracy, some pre-processing steps were
applied as described above in section `Preprocessing of the Images`. 

The adaptive histogram equalization step gave maximum accuracy improvement in the LeNet model. Initialization
weights also aided the accuracy improvement. At first weights were initialized to zeros which resulted in lower
accuracy values hence were initialized using Mean=0 and sigma=0.1 which helped in accuracy improvement. The maximum 
validation accuracy achieved using the steps above was around 93~94%.

However, maximum jump jump in accuracy was achieved when the input from lower convolutional layers were also
passed to fully connected layers as shown in `Modified CNN architecture` above. Initally, input from 2nd 
convolutional layer was passed to fully connected layers along with 3rd convolutional layer which bumped up
validation accuracy around 96%. Increasing the number of layers in the convolution process also helped increase
the accuracy to 97%. 

Finally, output from all the convolution layers were passed to fully connected layers which helped the model
to achieve 98% accuracy.

## Test model on new images
The above trained model was tested by downloading random images from the internet. The downloaded images were 
passed through algorithm for prediction and top 5 predictions were also returned by the algorithm.
##### Acquiring New Images :
Eight random images of german traffic signs were downloaded from internet. Initially images with minimal background 
(i.e  yellow or white background) were chosen to check the accuracy of the code. 

The code had no difficulty in predicting any of the images as expected. However, to add difficulty images 
with some background was added to the as shown in the images of No-entry, Yield sign and Pedestrian. 
The image pool is shown below

 ![alt text][image7]
 
##### Performance on New Images :
As described above the code had minimal difficulty in predicting images with no background as shown in the 
prediction below. This is also evident by the probability of the ground truth which exceeds other probability 
of other predictions by a considerable amount. _`Note : Probability is displayed on x-axis (in log format)`_ 
 
 ![alt text][image8]
 ![alt text][image10]
 ![alt text][image12]
 ![alt text][image13]
 ![alt text][image15]
 
 A second set of images with some background were tested with the same algorithm. This set did not show consistent 
 results. As seen below in the image of `pedestrian` sign which is predicted as `traffic signal` by the algorithm. 
 Looking at top probable predictions for the image shows pedestrian as third most probable sign. Hence, 
 algorithm is not able to recognize pedestrian sign. The yield sign shown below is predicted correctly only 
 around 70% of the time.   
 
 ![alt text][image11]
 ![alt text][image14]
 ![alt text][image9]
      
      
### Investigation for bad classification of the Images:

#### Visualization of feature maps
In order to check for the problem with traffic sign classification. I examined the images of the feature maps generated 
in convolutional layers for images in both the sets. A closer examination indicated that the features detected in the 
1st set of images are more well defined as compared to features maps in the 2nd set of images.

The image shown below is the image of feature maps for 1st and 2nd covolutional layers for Priority road sign which belonged
to 1st set of images. The feature maps in layer 1 shows well defined edges.

**Convoluition layer-1 feature maps - Priority road sign**
 ![alt text][image16]
 
**Convoluition layer-2 feature maps - Priority road sign**
 ![alt text][image17]

A similar feature map when plotted for the pedestrian sign shows that the network is not able to clearly distinguish 
between background and the traffic sign correclty.

**Convoluition layer-1 feature maps - Pedestrian road sign**
 ![alt text][image18]
 
**Convoluition layer-1 feature maps - Pedestrian road sign**
![alt text][image19]

#### Visualization of prediction scores:
Some other scores were also plotted to check how well model is working on certain labels as compared to others.
This is done by plotting precision, recall and F-score bar charts for both testing and validation sets. 

The image below shows the bar charts for precision, recall and F-scores for the testing dataset. A closer look 
at the charts shows that class Id-27 which corresponds pedestrian has low F-score on testing dataset. 
 ![alt text][image20]