
Designing a Convolutional Neural Network (CNN) model for the specified task involves several components and considerations.

**Data Preprocessing:**

The input images have a size of 224x224 pixels, and the goal is to predict three values: X, Y, and R. Before feeding the images into the model, normalization is applied to both the image tensor and the labels generated from the circle method. The normalization is performed with a mean of 0.5 and a standard deviation of 0.5.

**Model Architecture:**

1. **MobileNetV2 Backbone:**
   - This component serves as the feature extraction backbone and is based on the MobileNetV2 architecture. It takes an input image and resizes it to (224, 224) pixels.
   - The MobileNetV2 backbone is pretrained on ImageNet, providing a strong foundation for feature extraction.
   - The last classification layer of MobileNetV2 is replaced with a custom fully connected layer containing 256 output features.

2. **Regressor:**
   - The Regressor component consists of two fully connected layers.
   - It takes the output from the MobileNetV2 backbone and maps it to a 3-dimensional output, representing regression values for X, Y, and R.

3. **StarDetector:**
   - The StarDetector model is the overarching structure that combines the MobileNetV2 Backbone and the Regressor.
   - It utilizes the forward method to enable regression and predict the target output, which includes the coordinates X and Y, as well as the radius R.

**Loss Function:**

Given that this is a regression task, the Mean Squared Error (MSE) is chosen as the loss function. MSE measures the average squared difference between the predicted and true values, providing an effective means to evaluate the model's performance on the regression task.

In summary, the designed model utilizes a MobileNetV2 backbone for feature extraction, a Regressor for mapping features to regression outputs, and a StarDetector to combine these components for the overall prediction of X, Y, and R. The choice of MSE as the loss function aligns with the regression nature of the task.



**Intersection of Union IOU SCORE**

IOU score of 0.95 on 500 samples with a noise level of 0.5
