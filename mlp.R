#In command line:
#ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#brew install openblas
#brew install opencv


#cran=getOption("repos")
#cran["dmlc"]="https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
#options(repos = cran)
#install.packages("mxnet")
#install.packages("caret")

require(mxnet)

data=read.csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
#data=data.table::fread('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
data=data[complete.cases(data), ]

library(caret) #this package has the createDataPartition function
set.seed(123) #randomization`
#creating indices
trainIndex=createDataPartition(1:dim(data)[1],p=0.75,list=FALSE)
#splitting data into training/testing data using the trainIndex object


#define outcome & input
y=data[,'Churn']
d=data[,1:20]

#change outcome & input to fit in mxnet
yy=as.numeric(y)-1
dd=sapply(d,as.numeric)
#dd=data.matrix(scale(dd))

trainyy=yy[trainIndex]
testyy=yy[-trainIndex]
traindd=dd[trainIndex,]
testdd=dd[-trainIndex,]
traindd=data.matrix(scale(traindd))
testdd=data.matrix(scale(testdd,attr(traindd, "scaled:center"), attr(traindd, "scaled:scale")))



#construct MLP symbol
data1=mx.symbol.Variable("data")
fc1=mx.symbol.FullyConnected(data1, num_hidden=20)
act2=mx.symbol.Activation(fc1,  act_type="relu")
fc2=mx.symbol.FullyConnected(act2,  num_hidden=2)
softmax=mx.symbol.SoftmaxOutput(fc2)

devices=mx.cpu()
mx.set.seed(0)

#set model
model=mx.model.FeedForward.create(softmax, X=traindd, y=trainyy,
                                     ctx=devices, num.round=30, array.batch.size=40,
                                     learning.rate=0.1, momentum=0.9,eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))
#get probability matrix
preds = predict(model, testdd)
predict.test.df=data.frame(t(preds))
pp=predict.test.df

#get accuracy
pred.label = max.col(pp)-1
df.pred=data.frame(table(pred.label, testyy))
df.pred
df.pred$pred.label=as.numeric(as.character(df.pred$pred.label))
df.pred$yy=as.numeric(as.character(df.pred$testyy))
ind=which(df.pred$pred.label==df.pred$yy)
pred.accuracy=sum(df.pred[,3][ind])/sum(df.pred[,3])
cat("Prediction accuracy:","\n")
print(pred.accuracy)

#get graph flowsheet
graph.viz(model$symbol)

