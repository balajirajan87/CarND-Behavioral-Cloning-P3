# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Model_Arch.png "Model Visualization"
[image2]: ./images/org_image.jpg "Original Image from the central camera"
[image3]: ./images/flipped_image.jpg "Flipped image used for Image data augmentation"
[image4]: ./images/cropped_normalized_image.jpg "Image after preprocessing like Image cropping and normalization"
[image5]: ./images/Recovery_Drive_org_image.jpg "Recovery Image"
[image6]: ./images/Recovery_Drive_flipped_image.jpg "Recovery Image flipped for image Data Augmentation"
[image7]: ./images/Recovery_Drive_cropped_normalized_image.jpg "Recovery driving Image after preprocessing like Image cropping and normalization"
[image8]: ./images/lossimage.png "Loss Visualization"
[image9]: ./images/Side_cameras_org_image.jpg "image from the right camera"
[image10]: ./images/Side_cameras_flipped_image.jpg "flipped image from the right camera"
[image11]: ./images/Side_cameras_cropped_normalized_image.jpg "cropped and normalized image from the right camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

### Please refer to the code archived in [GitHub](https://github.com/balajirajan87/CarND-Behavioral-Cloning-P3.git)

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Model.py containing the pipeline script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 capturing the video from the front facing centre camera while driving autonomously.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My Sequantial Model first starts with Image Preprocessing Layer known as Lambda layer. This layer takes in the input image of size (160,320,3), and performs image normalization such that
the image data is mean centered to 0 and has standard deviation of 0.5. (model.py line: 96).
```sh
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```
Next i use a cropping2D layer for cropping of the top layer ( that has tries, mountains, skies) and the bottom layer (that has the car bonnet peaking out) that can potentially confuse the model) (model.py line: 97)
```sh
model.add(Cropping2D(cropping=((50,20),(0,0))))
```
Next i use a convolution layer with filter size 3x3 and i use 24 filters. I use the stride frequency of 2, to convert my input image of size (160,320,3) to an image of (80,160,24). I use "ELU" activation layer that introduces non linearity. (model.py line: 98).
```sh
model.add(Convolution2D(24,3,3,subsample=(2,2),activation="elu"))
```
Next i use an another convolutional layer with filter size 3x3 and here i employ 48 filters, with stride frequency of 2 to convert the input data of size 80,160,24 to an image of (40,80,48).I use "ELU" activation layer that introduces non linearity. (model.py line: 99). 
```sh
model.add(Convolution2D(48,3,3,subsample=(2,2),activation="elu"))
```
After this i halve the data by using the Maxpooling layer that uses filter size of 4x4 and a stride frequency of 4 that converts my input data of size (40,80,48) to a data of size (10, 20,48)(model.py line: 100).
```sh
model.add(MaxPooling2D(pool_size=(4, 4), strides=4))
```
Next i use the Dropout layer with 20% data being removed. This converts my (10,20,48) sized data to: (8,16,48) data(model.py line: 101).
```sh
model.add(Dropout(0.2))
```
Next i flatten the data by using the keras.Models Flatten() function call. This converts the (8,16,48) sized data to an array of size 6144.  (model.py line 102)
```sh
model.add(Flatten())
```
The rest of the model involves using three layers of Dense layers with "elu" activation that effectively converts the 6144 sized data to a single numeric data  (Model.py lines 103-105). 
```sh
model.add(Dense(50,activation="elu"))
model.add(Dense(10))
model.add(Dense(1))
```
#### 2. Attempts to reduce overfitting in the model

As explained in the previous section, the model contains dropout layers in order to reduce overfitting (model.py line: 101). Apart from this i also make sure that the data sets are well shuffled before splitting to training and validation sets ( model.py line: 27-28). The randomness in the training Data sets makes sure that the training is efficient and that the model can generalize well on the Test data that is presented.
```sh
samples_shuffled = shuffle(samples)
train_samples, validation_samples = train_test_split(samples_shuffled, test_size=0.1)
```
Also during training i make sure to stop the training when the 'mse' reaches a certain lower threshold using keras.callbacks() function call. i use a threshold value of 0.05 (5% loss). refer model.py lines: 82-92
```sh
# defining the loss_threshold for stopping the learning based on 'mse'.
LOSS_THRESHOLD = 0.05

#Callback class for dynamic stopping of learning.
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_mean_squared_error') < LOSS_THRESHOLD):
            print("\nReached %2.2f%% Loss, so stopping training!!" %(LOSS_THRESHOLD*100))
            self.model.stop_training = True

#Instantiation of the callback class defined above.
callbacks = myCallback()
```
and i use this callbacks function in my fit.generator function call as below:
```sh
model.fit_generator(train_generator,steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),epochs=20, verbose=1, callbacks=[callbacks])
```
This effectively stops my training at around 10 epochs and thereby prevents my model from overfitting.

#### 3. Model parameter tuning

I use an Adam Optimizer with a learn rate of 0.0001, and i configure the optimizer to work towards attaining a minimum 'mean squared error' loss. refer model.py lines: 106-108
```sh
opt = adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt,metrics=[metrics.mean_squared_error])
history_object = model.fit_generator(train_generator,steps_per_epoch=np.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=np.ceil(len(validation_samples)/batch_size),epochs=20, verbose=1, callbacks=[callbacks])
```
After several experiments i found that instead of using a default learn rate, selecting a lower Learn rate created a better Neural network models that could estimate the steering angles better.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road and within the lanes. I created training datas that consisted of lane centre driving (here i used continuous steering angles by using mouse keys without lifting them so that the datas were not mean centred with 0 steering angle), in both the Clockwise and Counter-Clockwise directions, and Recovery Driving that primarily focuses on collecting datas where the vehicle is driven from the sides towards the Lane centre and also driving on the Steep Curves. This Recovery Datas were collected so that we teach the model on how to behave when the centre camera sees a Curve or when the centre camera recognises that vehicle is coasting off the lane. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a simple but an efficient convolutional Neural network.

My first step was to use a convolution neural network model similar to that used by NVIDIA for their self driving car. Refer [link](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate since the model employed 5 convolutional layers with ReLU activations. I split the data set to training and validation set with 0.9 and 0.1 split, and i trained the model for 10 epochs but to my surprise i found that the model was not performing good and it deviated from the lanes where it was expeted to drive Straight ahead. with the first model, i good a pretty good low mse on training sets but the mse on validation set was slightly higher. This meant that the model was overfitting.

To combat the overfitting, I simplified the overall CNN model by removing three convolutional Layers. So the final model had the below layers in the mentioned order:
1. A preprocessing lambda layer that normalizes the image.
2. A cropping layer that crops off the unnecessary details in the image. 
3. Then two Convolutional layers each with 3x3 filters, and stride frequency of 2 and with "ELU" activation.
4. A MaxPooling layer with 4x4 filters, and with stride freq of 4.
5. A Dropout layer that drops 20% of the data.
6. A Flatten layer to convert the data from image form to an array form.
7. Three Dense Layers to produce a single output (i.e, the steering angle)

I used a Generator routine to batch the entire dataset and get the samples in batches at the runtime. Please refer the Model.py file lines:  34-72. I used a batch size of 128. So the Generator processes 128 samples each time and appends the data to the data List that would be later used to train the model. 
```sh
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.4
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    source_len = len(source_path.split('/'))
                    filename = source_path.split('/')[4:source_len]
                    current_path = './'
                    for j in range(len(filename)):
                        if (j != len(filename)-1):
                            current_path += filename[j]
                            current_path += '/'
                        else:
                            current_path += filename[j]
                    image = cv2.imread(current_path)
                    images.append(image)
                    images.append(cv2.flip(image,1))    #flipping the image using cv2
                steering_centre = float(batch_sample[3])
                angles.append(steering_centre)
                angles.append(steering_centre * (-1.0)) #negating the steering angle for centre camera
                steering_left = steering_centre + correction    #steering angle for image obtained from left camera
                angles.append(steering_left)
                angles.append(steering_left * (-1.0))   #negating the steering angle for left camera
                steering_right = steering_centre - correction   ##steering angle for image obtained from right camera
                angles.append(steering_right)
                angles.append(steering_right * (-1.0))  #negating the steering angle for right camera

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
```
Within the Generator routine i made sure to augment the data by using the images from the left and the right cameras apart from the central one. When using the left camera image i made sure to reduce the steering angle and when using the right camera model i made sure to increase the steering angle. So the Correction factor for the steering angle was 0.4. Please refer to the above mentioned code snippet proided. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like after the bridge where the vehicle had to steer to the left, the vehicle drove straight to the muddy area. This was probably because the network got confused due to the abscence of the lane marking to the right and hence the vehicle drove straight into the muddy area. Again after the left curve, there was another curve to the right where inspite of the presence of Lane markings the vehicle drove straight into the Waters. To improve the driving behavior in these cases, I specifically collected the training datas (in the dataset you can find Recovery_datasets) that consisted of driving on the curves and i repeated the process of data collection for three times for each curves.

With the datas collected, i employed the same data generator and training routine as explained in the previous sections, and at the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 95-105) consisted of a convolution neural network with the following layers and layer sizes 
1. A preprocessing lambda layer that normalizes the input image of size (160,320,3)
2. A cropping layer that crops off the unnecessary details in the image. (top 50 rows and bottom 20 rows)
3. A Convolutional layer with 24 (3x3) filters, and stride frequency of 2, and with "ELU" activation to create an output data of size (80,160,24)
4. A Convolutional layer with 48 (3x3) filters, and stride frequency of 2, and with "ELU" activation to create an output data of size (40,80,48)
4. A MaxPooling layer with 4x4 filters, and with stride freq of 4. This reduces your data to (10,20,48)
5. A Dropout layer that drops 20% of the data. The data that is output from this layer is of size(8,16,48)
6. A Flatten layer to convert the data from image form to an array form. so the ouput data from this layer is of an array form that has size (8*16*48)
7. One Dense layer with "ELU" Activation to reduce the 6144 size data to 50.
8. One Dense layer to reduce the 50 sized data to 10
9. Final dense layer to reduce the 10 sized data to 1.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here i used mouse to steer carefully around the track without lifting the mouse key. This ensured that the steering angle between each and every sample was continuous, and this ensured that my steering training data does not have too many samples collected on Steering angle of 0. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the lane centre in case the vehicle steers to the side of the road. These images show what a recovery looks like:-

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles, since this data augmentation process would help my car to recover more from the road sides to the lane centre whenever the centre camera sees a similar image during the testing / deployement. For example, here is an image that has then been flipped:

![alt text][image3]

I also applied the above explained Data Augmentation process for the images collected for Recovery driving. Below you can find an image that was being collected while driving from side towards the lane centre, and the image being flipped.

![alt text][image5]
![alt text][image6]

The Image datas were further Augmented by collecting the data from the left and Right cameras and performing the above mentioned Image processing. Please find below the image from the Right Camera, and the flipped image.

![alt text][image9]
![alt text][image10]

After the collection process, I had 13,277 number of data points. I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 10 as evidenced by the model being stopped by the Keras Callbacks when the mse reached below 0.05 (i.e, 5%). I used an adam optimizer so that manually training the learning rate wasn't necessary.