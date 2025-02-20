# **Behavioral Cloning** 

The output video of the model is visible [here](./data/output_video_2.mp4)
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/figure_loss.png "Loss / Epoch"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `src/model.py` containing the script to create and train the model
* `src/drive.py` for driving the car in autonomous mode
* `data/model.h5` containing a trained convolution neural network 
* `data/output_video_2.mp4` containing the video of the final simulation
* `README.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a five convolution neural network with 5x5 filter sizes and depths between 24 and 64 (`model.py` lines 154-158). The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 149-153). 

| Layer         		|     Description	        					            |  Activation         		|
|:---------------------:|:---------------------------------------------------------:| :------------------------:|
| Input         		| 160x320x3 RGB image 							            |                           |
| Lambda             	| Normalization to intervale [-0.5, 0.5]	                |                           |
| Crop					| top 70 px and bottom 6px to reduce distractions           |                           |
| Lambda              	| Flatten the 3 channels (0.21 * R + 0.72 * G + 0.07 * B)   |                           |
| Convolution 5x5	    | 2x2 stride, same padding, number filters = 24         	| RELU                      |
| Convolution 5x5	    | 2x2 stride, same padding, number filters = 36         	| RELU                      |
| Convolution 5x5	    | 2x2 stride, same padding, number filters = 48         	| RELU                      |
| Convolution 5x5	    | 2x2 stride, same padding, number filters = 64         	| RELU                      |
| Convolution 5x5	    | 2x2 stride, same padding, number filters = 64         	| RELU                      |
| DROPOUT   			|           		                            			|                           |
| FULLY CONNECT LAYER	| output dimension = 100 					                |                           |
| FULLY CONNECT LAYER	| output dimension = 50 					                |                           |
| FULLY CONNECT LAYER	| output dimension = 10 					                |                           |
| OUTPUT            	| output dimension = 1			    	                    |                           |

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers (with 0.9 keep probability) in order to reduce overfitting (`model.py` lines 159). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 25-137). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 167-175).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, using multiple cameras and flipping the images horizontally.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to:

As the model proposed by NVIDIA autonomous driving team is a good starting point, I initially adopted it.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Therefore I introduced a single dropout layer after the convolutional layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected additional driving data in order to teach the vehicle out to recover from left and right side. Additionally for each original camera image, I augmented the data by adding  also their horizontally flipped version. This came with an additional challenge:

- Central camera -> flip horizontally and multiply steering angle by -1 (`model.py` line 93-97 and function `model.py -> extend()` in line 38-67)
- Left camera -> flip horizontally and add the steering angle of the right camera (same as that of the central camera plus a correction factor) multiplied by -1 (`model.py` line 105-106 and function `model.py -> extend()` in line 38-67)
- Right camera -> flip horizontally and add the steering angle of the left camera (same as that of the central camera plus a correction factor) multiplied by -1(`model.py` line 109-110 and function `model.py -> extend()` in line 38-67)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 146-164) is given the the table in paragraph 1.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I started with the initial provided data and added some recovering from the sides behavior.


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it find itself close to the borders. An example can be seen in the final video.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by
simulations and visible in the following figure

![alt text][image1]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

The output video of the model is visible [here](./data/output_video_2.mp4)