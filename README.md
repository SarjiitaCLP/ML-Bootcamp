#### Multivariate Linear Regression

### - Step 1: Training Dataset preprocessing
1. The training dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.

### - Step 2: Defining the class `lin_reg` which contains all the methods required to train the dataset
1. The method `hypothesis` calculates the weighted sum of all features (theta0.x0 + theta1.x1 + theta2.x2 +...), which is the predicted value.
2. The method `error` calculates the difference between the predicted output and actual label. Error = prediction - actual
3. The method `grad` calculates the gradient of the cost function that we have to optimise, and multiplies it with the learning rate. gradient_i = (learning rate/number of features) * (error_i * train_xi)
4. The method `update_theta` updates the value of weights after each iteration. weight:= weight - gradient 
5. The method `rmse` calculates the loss in each epoch. First, it calculates the mean squared error using the formula: MSE = (summation of squares of delta_i)/(number of features). Next, it calculates the square root of MSE to obtain RMSE.
7. The method `train` does the job of training the dataset. It is run `epoch` number of times. It contains a list called `rmse_list` that stores the loss after each epoch. First, the method `hypothesis` is called to predict a value using the `theta` array initialized by random numbers. Next, the method `error` calculates the error in our prediction and `grad` gives the value of gradient which we need to subtract from `theta` using `update_theta`. The loss in that epoch is also calculated by calleing the method `rmse`, the value of which is stored in `rmse_list`. The same procedure is done until the `for` loop ends. It returns the array `theta` and the array `rmse_list`.

### - Step 3: Using the class `lin_reg` to train the dataset
1. The array `theta` is initialized with random numbers.
2. The method `train` is called from the `lin_reg` class and '0.001' & '50' are passed as the values of learning rate and number of epochs respectively.
3. A graph is plotted between `epoch` and `rmse_list`, which shows the behaviour of training loss with iterations.
4. The predicted values for training dataset are calculated using the `theta` values.

### - Step 4: Test Dataset Preprocessing
1. The test dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.

### - Step 5: Prediction of values on Test Dataset
1. Predictions are made on the test dataset using the `theta` values and the RMSE is also calculated.


#### Multiclass Logistic Regression

### - Step 1: Training Dataset preprocessing
1. The training dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.

### - Step 2: Enable One-hot-classification
1. First it is checked how many unique classes are present in the label and stored in the list `u`. The length `clas` of `u` denotes the number of classes. Here, clas=10
2. One-hot-classification is performed. A 2D numpy array named `one_hot_label` of size (19999 x 10) is created which stores the respective class values in the form of '0' & '1' for each rows from training dataset.

### - Step 3: Defining the class `log_reg` which contains all the methods required to train the dataset
1. The method `hypothesis` calculates the weighted sum of all features (theta0.x0 + theta1.x1 + theta2.x2 +...).
2. The method `sigmoid` calculates the output of the sigmoid function. Sigmoid function is: 1/(1+e^(-x))
3. The method `sigmoid_grad` calculates the gradient of the sigmoid function. Gradient: sigmoid_output/(1 - sigmoid_output)
4. The method `error` calculates the difference between the predicted output and actual label. Error = prediction - actual
5. The method `train` does the job of training the dataset. It is run independently for each class values from 0 to 10; `epoch` number of times for each class value. First, the method `hypothesis` and `sigmoid` are called to predict a value using the `theta` array initialized by random numbers. Next, the gradients are calculated and stored in `sigmoid_der`. The errors between predicted values & the class values in `one_hot_label` are obtained by using the method `error`. At the end, the gradient of the cost function is calculated by multiplying the transpose of input `train_x`, `sigmoid_der` and `err`. Mathematically, gradient_i = (predicted - label) * sigmoid_output * (1 - sigmoid_output) * input
6. The value of theta is updated using the formula: theta:= theta - learning rate * gradient 

### - Step 4: Using the class `log_reg` to train the dataset
1. The array `theta` is initialized with random numbers.
2. The method `train` is called from the `log_reg` class and '0.001' & '500' are passed as the values of learning rate and number of epochs respectively.

### - Step 5: Test Dataset Preprocessing
1. The test dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.
5. One-hot-classification is enabled in the same way as for feature dataset.

### - Step 6: Prediction of values on Test Dataset
1. Predictions are made on the test dataset using the `theta` values and the accuracy is also calculated.


#### kNN:

### - Step 1: Preprocessing of Training dataset
1. The training dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.

### - Step 2: Test Dataset Preprocessing
1. The test dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. The bias column is added at the beginning of the feature set which contains only the value '1'.
4. Both feature and label dataframes are converted to numpy array.

### - Step 3: Defining the class `knn` which contains all the methods required to train the dataset
1. The method `euclidian_distance` calculates the Euclidian distance function to get distance between a test row and all training rows. 
2. The method `sort` sorts out 'k' points in the training dataset that are closest to the test dataset. It stores the indices of those 'k' closest points.
3. The method `class_value` replaces the sorted indices with their actual class values corresponding to those indices.
4. The method `class_label` counts the maximum occurring class value for a given test datapoint.

### - Step 4: Predict the class values and calculate the accuracy of prediction
1. The class values are predicted by calling the methods from the class `knn`.
2. The accuracy is calculated.


#### Neural Network

### - Step 1: Training Dataset preprocessing
1. The training dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. Both feature and label dataframes are converted to numpy array.

### - Step 2: Enable One-hot-classification
1. First it is checked how many unique classes are present in the label and stored in the list `u`. The length `clas` of `u` denotes the number of classes. Here, clas=10
2. One-hot-classification is performed. A 2D numpy array named `one_hot_label` of size (19999 x 10) is created which stores the respective class values in the form of '0' & '1' for each rows from training dataset.

### - Step 3: Defining the class `ann` which contains all the methods required to train the dataset
1. The method `hypothesis` calculates the weighted sum of all features (theta0.x0 + theta1.x1 + theta2.x2 +...).
2. The method `sigmoid` calculates the output of the sigmoid function. Sigmoid function is: 1/(1+e^(-x))
3. The method `sigmoid_grad` calculates the gradient of the sigmoid function. Gradient: sigmoid_output/(1 - sigmoid_output)
4. The method `softmax` calculates the output of the softmax function. Softmax function is: (e^x)/summation of all e^(x)
5. The method `train` does the job of training the dataset. It is run `epoch` number of times. Firstly, the feedforward phase is performed. During the feedforward between input layer to hidden layer, sigmoid function does the job of calculating the prediction and stores them in `ah`. Between hidden layer and output layer, softmax function calculates the prediction and stores them in `ao`. During the backpropagation from output layer to hidden layer, the following formula is used: gradient= ah * (predicted - label). During the backpropagation from hidden layer to input layer, the following formula is used: gradient= (ao - label) * wo * sigmoid_output * (1 - sigmoid_output) * input. The weights and biases for both hidden layer and output layer are updated.

### - Step 4: Using the class `ann` to train the dataset
1. Number of hidden nodes = 4
2. The contents of `clas` has been copied into `output_labels`.
3. `wh` and `bh` are weights and bias respectively, for hidden layer; wo and bo are weights and bias respectively, for output layer. All are initialized with random values.
4. The method `train` is called from the `ann` class and '0.0000001' & '50' are passed as the values of learning rate and number of epochs respectively.

### - Step 5: Test Dataset Preprocessing
1. The test dataset is split into features and label.
2. The feature set is normalized by dividing by 255.
3. Both feature and label dataframes are converted to numpy array.
4. One-hot-classification is enabled in the same way as for feature dataset.

### - Step 6: Predict outputs on test dataset
1. Predictions are made on the test dataset using the weight values and the accuracy is also calculated.
