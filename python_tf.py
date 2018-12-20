
# coding: utf-8

# # Netural Networks: Multilayer Perceptron Models - Tensorflow
# 
# ## *Group 18: Ming Gao, Zhe Yin, Zizhao Zhang*
# 
# ### Introduction | <a href="./introduction.html">Home</a>
# 
# ### Introduction & General Sense: 
# The neural network works in terms of a framework for many different machine learning algorithms to work together and process complex data inputs. The neural network statistical data modeling tool is where the complex relationships between inputs and outputs are modeled or sepecific confounded patterns are found to perform mainly tasks like clustering, classification, pattern recognition and other unsupervised machine learnings.
# 
#  The below process and simulation are build via the frameworks of Tensorflow in python 3.6. The framework of neural networks mimicks the nerons in biological sense. In that each neturon within the network consists of sessions, graphs, placeholders, and variables as key building blocks. I will implement these through the tool of tensorflow in the below process.
# ***
#  <font color=purple>Below is a demonstration of the basic model structure of 1 hidden layer Artificial neural network</font>
# ![basic structure](https://tgmstat.files.wordpress.com/2013/05/neural_network_example1.png)
# 

# ### How to install tensorflow in the current: working environment:
# If you have not installed tensorflow in the current working station system, you have the below two options using pip:
# 
# **FIRST UPDATE PIP:**   <font color=red>*pip install --upgrade pip*</font>
# ***
# 
# **VIRTUALENVIRONMENT INSTALL:**   <font color=red>*pip install --upgrade tensorflow*</font>
# ***
# 
# **SYSTEM INSTALL:**   <font color=red>*pip3 install --user --upgrade tensorflow  # install in $HOME*</font>
# ***
# 
# **VERIFY INSTALLATION:**  <font color=red>*python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"*</font>
# 
# 
# 
# 

# Below are the required packages for the simulation of the Multilayer Perceptron Models using tensorflow in Python 3
# 
# Make sure that you can load them before trying to run the examples on this page.
# 
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
#ingnore the warnings for html output
import warnings
warnings.filterwarnings('ignore')


# ### description of the dataset for modeling: 
# 
# The selected dataset can be found by the below link: https://www.kaggle.com/blastchar/telco-customer-churn Each row within the Telcom Customer Churn dataset represents a customer, and each column contains customer’s attributes described on the column Metadata.The raw data contains 7043 rows (customers) and 21 columns (features). <font color=blue>The “Churn” column is our desired response.</font> All other details of the other featuers of the datasets can be found via the above link.
# 

# ### Dataset set-up and data cleaning:
# 

# In[2]:


#import the dataset as panda dataframes
data = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
#select the first 20 columns as X, factorize the categorical elements
X = data.iloc[:, :20].astype('category')
for i in range(X.shape[1]):
    X.iloc[:, i] = X.iloc[:, i].cat.codes
#factor "Churn" our response as 0,1
y = data.iloc[:, -1].astype('category').cat.codes
#set up 
input_size = X.shape[1]
hidden_size = 20
num_classes = 2


# below is the first 5 rows of the dataset

# In[3]:


X.head()


# Below is the set up before implementing into the tensorflow tool

# In[4]:


#ingnore the warnings for html output
import warnings
warnings.filterwarnings('ignore')
#set up testings sets and training sets(porportion of 75%)
traind, testd, trainy, testy = train_test_split(X, y, train_size=0.75)
traindd = traind.values;
testdd = testd.values
#convert categorical variable into dummy/indicator variables
trainy = pd.get_dummies(trainy);
testy = pd.get_dummies(testy)
trainyy = trainy.values;
testyy = testy.values
#set the scaler to minmax
#fit and transform 

data_scaler = preprocessing.MinMaxScaler()
traindd = data_scaler.fit_transform(traindd);
testdd = data_scaler.fit_transform(testdd)


# In[5]:


traindd[:3,]


# In[6]:


trainyy[:3]


# ### set up for the neura network frameworks: 
# 
# The number of categories the model is choosing from: variable- <font color=purple> *numClasses* </font> set equal to 2
# ***
# A 28x28 image will have 784 total pixel values : variable - <font color=purple> *inputSize* </font> set equal to 20
# ***
# The number of hidden units this one layer NN will have : <font color=purple> *numHiddenUnits* </font> set equal to 20, the number of hidden unites within each layer (which we have two) should equals to the number of columns of X
# ***
# The number of times the training loop is run : <font color=purple> *trainingIterations* </font> set equal to 30, by the end we will have 30 steps within each training process in the session
# ***
# The number of images/data points we feed in one training batch: <font color=purple> *batchSize* </font> set equal to 10 since our dataset is not huge, setting batch sizes as 10 should be appropriate
# 

# In[7]:


#The number of categories the model is choosing from: variable- numClasses set equal to 2
numClasses = 2  
#A 28x28 image will have 784 total pixel values : variable - inputSize set equal to 20
inputSize = traindd.shape[1] 
#The number of hidden units this one layer NN will have : numHiddenUnits set equal to 20
numHiddenUnits = 20  
#The number of times the training loop is run : trainingIterations set equal to 30
trainingIterations = 30  
#The number of images we feed in one training batch: batchSize set equal to 10
batchSize = 40


# Now we make sure each time running the code, the graph refreshes itself. On the other hand set the tensorflow placeholders type and shape.

# In[8]:


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, inputSize])
y = tf.placeholder(tf.float32, shape=[None, numClasses])


# Now set the Ws and the Bs for the alpha function of the two hidden layers and the intermediate matrix

# In[9]:


W1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numClasses])


# Now define the alpha function within the hidden layers consists the matrix multiplication, and define the final output

# In[10]:


hiddenLayerOutput = tf.matmul(X, W1) + B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
finalOutput = tf.matmul(hiddenLayerOutput, W2) + B2


# In[11]:


#ingnore the warnings for html output
import warnings
warnings.filterwarnings('ignore')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#now we passes in the operation into the session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(trainingIterations):
        idx = np.random.choice(traindd.shape[0], batchSize, replace=False)
        batchInput = traindd[idx]
        batchLabels = trainyy[idx] #feed batchinput into the feed dictionary
        _, _ = sess.run([opt, loss], feed_dict={X: batchInput, y: batchLabels})
        if i % 1 == 0:
            trainAccuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
            print ("step %d, training accuracy %g" % (i+1, trainAccuracy))
    # test
    acc = accuracy.eval(feed_dict={X: testdd, y: testyy})
    print("testing accuracy: {}".format(acc))

