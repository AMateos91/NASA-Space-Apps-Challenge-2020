## NASA Space Apps Challenge Project##

# #

# Dataset loading 

library(ranger)
library(caret)
library(data.table)
library(CRAN)
library(Rtsne)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
url<- "https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/2018_01/*/auxil/ni*.csv"
spacedata<-read.csv(url)
head(spacedata)

# Data exploration / cleaning #

dim(space_data)
head(space_data, 130)
tail(space_data, 130)

table(space_data$Class)
summary(space_data$Amount)
names(space_data)
var(space_data$Amount)
sd(space_data$Amount)

space_data %>%
     mutate(id=1:n(nrows)),
     mutate(Class=as.Class.Integer)

names(space_data) = gsuv("Kind", "Distance", "Time")

# Data wrangling #

head(space_data)
space_data$Amount=scale(space_data$Amount)
NewData=space_data[,-c(1)]
head(NewData)

tsne_out <- Rtsne(as.matrix(select(space_data)),
		  pca = FALSE ,
		  theta = 0.35 ,
		  verbose = TRUE ,
		  max_iter = 2500 ,
		  Y_init = NULL ,)

# Data modeling #

set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)

# Logistic regression model #

Logistic_Model= glm(Class~.,test_data,family=binomial())
summary(Logistic_Model)
plot(Logistic_Model)
lr.predict <- predict(Logistic_Model,train_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, color = "blue")

# Decision Tree model #

decisionTree_model <- rpart(Class ~ . , space_data, method = 'class')
predicted_val <- predict(decisionTree_model, space_data, type = 'class')
probability <- predict(decisionTree_model, space_data, type = 'prob')
rpart.plot(decisionTree_model)

# Artificial Neural Network #

library(neuralnet)
ABE_model =neuralnet(Class~.,train_data,linear.output=FALSE)
plot(ABE_model)
	
predABE=compute(ABE_model,test_data)
resultABE=predABE$net.result
resultABE=ifelse(resultABE>0.5,1,0)

# Gradient boosting #

library(gbm, quietly=TRUE)
	
# Get the time to train the GBM model

system.time(
  model_gbm <- gbm(Class~.
                      , distribution = "bernoulli"
                      , data = rbind(train_data, test_data)
                      , n.trees = 500
                      , interaction.depth = 3
                      , n.minobsinnode = 100
                      , shrinkage = 0.01
                      , bag.fraction = 0.5
                      , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
                      )
  )
# Determine best iteration based on test data #

gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort = TRUE)

#Plot the gbm model #

plot(model_gbm)

# Plot and calculate AUC on test data

gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, color = "green")
print(gbm_auc)


  
  
  
