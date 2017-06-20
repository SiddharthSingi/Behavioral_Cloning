# Behavioral_Cloning


#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter and 5x5 filter sizes and depths between 24 and 64 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to simulate the car steering off the road and recovering back. I used the left and right images to create more data and used a correction factor of 0.2 for the left and right images.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to be able to correctly predict the steering angle using the images taken from the camera. This requires the layers to be able to capture the curvature of the road correctly which is why I started with a few layers of convolutional layers

My first step was to use a convolution neural network model similar to the NVIDIA End-to-End Deep Learning network, I thought this model might be appropriate because it was able to correctly predict steering angles with only the use of deep learning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there was a Dropout() layer after every Dense() layer.

Then I added 'Relu' activations after every convolutional layer

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially on areas where there were no lane lines, to improve the driving behavior in these cases, I drove around the edge of the same area and recovered back to the center. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:


| Layers        | Parameters           |
| ------------- |:-------------:|
| Lamda                            | Normalizing between (-0.5,0.5)   |
| Cropping2D                       | top=70, bottom=25                |
| Convolution2D                    |  Filter size = (5x5x24)          |
| Activation                       |  'Relu'                          |
| Convolution2D                    |  Filter size = (5x5x36)          |
| Activation                       |  'Relu'                          |
| Convolution2D                    |  Filter size = (5x5x48)          |
| Activation                       |  'Relu'                          |
| Convolution2D                    |  Filter size = (5x5x64)          |
| Activation                       |  'Relu'                          |
| Flatten                          |    -                             |
| Dense                            | Output=100                       |
| Dropout                          | keep_prob= 0.5                   |
| Dense                            | Output=50                        |
| Dropout                          | keep_prob= 0.5                   |
| Dense                            | Output=10                        |
| Dropout                          | keep_prob=  0.5                  |
| Dense                            | Output= 1                        |



####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center_2016_12_01_13_31_13_786](https://user-images.githubusercontent.com/26694585/27359730-0d7691ac-563b-11e7-8559-094007da44fa.jpg)


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![center_2017_06_21_03_26_34_295](https://user-images.githubusercontent.com/26694585/27359793-54727bde-563b-11e7-9735-8facc0e4d382.jpg)
![center_2017_06_21_03_26_35_126](https://user-images.githubusercontent.com/26694585/27359797-56ce188e-563b-11e7-8c19-e7f1e51772cf.jpg)
![center_2017_06_21_03_26_36_899](https://user-images.githubusercontent.com/26694585/27359803-5f1646ec-563b-11e7-9cec-6cbfec3c33d7.jpg)



After the collection process, I had 13396 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by by the model.ipynb output. I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Conclusion

This project taught me how to use deep learning for regression of steering angles. It was a lot of fun training the data myself and then looking at the model slowly learn how to drive. I now have a better understanding on how self driving cars learn in real life.

#### Improvements
To improve the model more data augmentation techniques could be implemented like changing properties of the images like brightness, lighing conditions, visibility etc. It will also be very helpful to use the other track along with the speed element to train the model to be more flexible to unseen driving conditions.
