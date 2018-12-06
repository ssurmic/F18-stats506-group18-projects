## Multilayer Perceptron (R with MXNet API)
##
## The Telco Customer Churn dataset can be downloaded from kaggle with url below:
##  https://www.kaggle.com/blastchar/telco-customer-churn
## In particular, see the csv data file, the.md file in our repo
##
## Author: Zhe Yin
## Updated: Nov 24, 2018


# Installing packages "mxnet" and "caret"
# Execute the R code below first. If nothing goes wrong, then you are good to go. 
# Otherwise you may need to execute the bash code to fix the problem.
# In R:
#cran=getOption("repos")
#cran["dmlc"]="https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
#options(repos = cran)
#install.packages("mxnet")
#install.packages("caret")


# In command line:
# if you've already installed Homebrew, openblas and opencv, you can just skip the following three lines of code
#ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#brew install openblas
#brew install opencv

# skip following two lines if your openblas and opencv are up-to-date
#brew upgrade openblas
#brew upgrade opencv

#ln -sf /usr/local/opt/openblas/lib/libopenblasp-r0.3.3.dylib /usr/local/opt/openblas/lib/libopenblasp-r0.3.1.dylib



#Steps:

# 1. Load the packages and data
require(mxnet) # this package enables us to train neural network model
library(caret) # the createDataPartition function would allow us to do cross validation

churn = read.csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn = churn[complete.cases(churn), ]

# 2. Define the output and input, and change them from categorical data into numerics to fit in mxnet
op = churn[,'Churn']
op = as.numeric(op) - 1
ip = churn[,1:20]
ip = sapply(ip, as.numeric)
head(ip)
head(churn)
churn[,21]

# Creating indices, the trainIndex object, and use it to split data into training and test datasets
set.seed(123) # randomization that controls the random process in createDataPartition
trainIndex = createDataPartition(1:dim(churn)[1], p = 0.75, list = FALSE)
train_op = op[trainIndex]
test_op = op[-trainIndex]
train_ip = ip[trainIndex, ]
test_ip = ip[-trainIndex, ]
train_ip = data.matrix( scale(train_ip) )
test_ip = data.matrix(   scale( test_ip, attr(train_ip, "scaled:center"),
attr(train_ip, "scaled:scale") )   )

# 3. Train the model in two steps
#    3a. Configure the model using the symbol parameter
#        Here we configure a neuralnetwork with two hidden layers,
#        where the first hidden layer contains 20 neurons and the second contains 2 neurons
data1 = mx.symbol.Variable("data")
fc1 = mx.symbol.FullyConnected(data1, num_hidden = 20)
act2 = mx.symbol.Activation(fc1, act_type = "relu")
fc2 = mx.symbol.FullyConnected(act2, num_hidden = 2)
softmax = mx.symbol.SoftmaxOutput(fc2)
#    3b. Create the model by calling the _model.FeedForward.create()_ method, 
#        you'll see the process of model training below:
devices = mx.cpu()
mx.set.seed(0)
# create a MXNet Feedorward neural net model with the specified training
model =
  mx.model.FeedForward.create(softmax, # the symbolic configuration of the neural network
                              X = train_ip, # the training data
                              y = train_op, # optional label of the data
                              ctx = devices, # the devices used to perform training (GPU or CPU)
                              num.round = 30, # the number of iterations over training data
                              array.batch.size = 40, # the batch size used for R array training
                              learning.rate = 0.1,
                              momentum = 0.9,
                              eval.metric = mx.metric.accuracy, # the evaluation function on the results
                              initializer = mx.init.uniform(0.07), # the initialization scheme for parameters
                              epoch.end.callback = mx.callback.log.train.metric(100) ) # the callback when one
                                                                                   # mini-batch iteration ends                                                                                              mini-batch iteration ends

# 4. Make a prediction and get the probability matrix, then calculate the accuracy rate
# make a prediction use the model trained
preds = predict(model, test_ip)
predict_test_df = data.frame(t(preds))
pred_test = predict_test_df
pred_label = max.col(pred_test) - 1
df_pred = data.frame( table(pred_label, test_op) )
# get the probability matrix
#df_pred
knitr::kable(df_pred)

df_pred$pred_label = as.numeric( as.character(df_pred$pred_label) )
df_pred$op = as.numeric( as.character(df_pred$test_op) )
# get the index where the prediction is correct
ind = which( df_pred$pred_label == df_pred$op )
# calculate the accuracy rate
pred_accuracy = sum(df_pred[,3][ind]) / sum(df_pred[,3])
print(pred_accuracy)

# 5. To get an idea of what is happening, view the computation graph from R
graph.viz(model$symbol)
