# Deep Neural Network Classifier
Deep Neural Network Classifier for the Win/Linux/OSX platform based on the GTK# Framework

**About Page**

![About Page](/Screenshots/about.png)

Deep Neural Network Classifier software uses a multilayer artificial neural network architecture. This architecture is comprised of an input layer, one or several hidden layers, and the output layer.

**Data Page**

![Data Page](/Screenshots/data.png)

Training and Test sets from a csv/text file can be loaded provided, you indicate the correct delimiter. Some network parameters are guessed automatically but can be modified. When loading training set data for classification, the last column in each line in the file is assumed to be the classification category. 0 is not counted as a classification category but in scenarios involving binary classification (0 or 1) it is handled automatically. 

**Training Page**

![Training Page](/Screenshots/training.png)

On this page, you can set several parameters that affects the training process. By default, the network utilizes only a single hidden layer. To use a deep neural network, set the number of hidden layers to a higher value, e.g. > 3.

This is where the actual training process happens. Learning rate refers to the rate or speed at which the network 'learns' or reconfigures itself by changing the interconnection strenghts between the nodes in each layer. In each iteration of the training, the difference between the network's current output and the expected output is measured (Error/Cost function). Two cost functions are provided, the default (cross entropy), and the L2 error. Tolerance is related to the mimimum value of the Error to consider the network 'trained' enough to stop.

Epochs refers to the maximum number of iterations to run the training. 

You can freely start/stop/continue the training process anytime. Training is performed during idle mode so you can freely move between pages. 

You do not need a test set to train the neural network. However, if a test set was loaded, it will automatically proceed to the classification step once training is completed. Classification threshold is the minimum value a test data point needs to score in order to be classified into a category. The default classification threshold is 50 but it can be set to values 1-100. For stricter classification a threshold of 90 (or higher) is recommended.

Once training is completed, you can enter new data points in the 'Test set' box and click on the 'Classify' button to classify them. If you change the number of hidden nodes, learning rate, or tolerance, you must retrain the network.

You can use the included the powerful Fmincg optimizer (C.E. Rasmussen, 1999, 2000 & 2001) to speed up the training process and obtain better classification performance. This is available via a check box near the bottom right corner of the Training page. You can use the L2 error instead of the default one to compare against the Tolerance during training. However, at the moment it performs poorly when using the Fmincg optimizer.

**Network Page**

![Network Page](/Screenshots/network.png)

Finally, trained network parameters can be saved and loaded (in JSON format) for use in future classification tasks or to provide a better starting point for training. Use the hidden layer selector to switch between hidden layers.

**Plot Page**

![Plot Page](/Screenshots/plot.png)

On this page, you can visualize the trained network's output on data sets that have only two (2) features. You can plot the classification output or the output plus contour curves showing the classification boundaries.

# Platform

Deep Neural Network Classifier software has been tested on Linux, OSX (soon), and Windows platforms.
