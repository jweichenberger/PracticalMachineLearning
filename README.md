# PracticalMachineLearning 

<h2>  Introduction </h2> 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

<h2> Data Preprocessing </h2> 

  _library(caret)_
  
  _library(rpart)_
  
  _library(rpart.plot)_
  
  _library(randomForest)_
  
  _library(corrplot)_

<h2> Data Loading </h2> 

  _trainFile <- "./data/pml-training.csv"_
  
  _testFile  <- "./data/pml-testing.csv"_
  
<h2> Data Reading </h2> 

After downloading the data from the data source, we can read the two csv files into two data frames.

  _trainRaw <- read.csv("./data/pml-training.csv")_
  
  _testRaw <- read.csv("./data/pml-testing.csv")_
  
  _dim(trainRaw)_
  
  _dim(testRaw)_

The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The "classe" variable in the training set is the outcome to predict.

<h2> Data Cleanup </h2> 

In this step, we will clean the data and get rid of observations with missing values as well as some meaningless variables.

  _sum(complete.cases(trainRaw))_
  
First, we remove columns that contain NA missing values.

  _trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0]_
  
  _testRaw <- testRaw[, colSums(is.na(testRaw)) == 0]_ 
  
Next, we get rid of some columns that do not contribute much to the accelerometer measurements.

  _classe <- trainRaw$classe_
  
  _trainRemove <- grepl("^X|timestamp|window", names(trainRaw))_
  
  _trainRaw <- trainRaw[, !trainRemove]_
  
  _trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]_

  _trainCleaned$classe <- classe_

  _testRemove <- grepl("^X|timestamp|window", names(testRaw))_

  _testRaw <- testRaw[, !testRemove]_

  _testCleaned <- testRaw[, sapply(testRaw, is.numeric)]_
  
Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

<h2> Data Slicing </h2> 

Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps.

  _set.seed(22519) # For reproducibile purpose_
  
  _inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)_
  
  _trainData <- trainCleaned[inTrain, ]_
  
  _testData <- trainCleaned[-inTrain, ]_

<h2> Data Modeling </h2> 

We fit a predictive model for activity recognition using Random Forest algorithm because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use 5-fold cross validation when applying the algorithm.

  _controlRf <- trainControl(method="cv", 5)_
  
  _modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)_
  
  _modelRf_
  
Then, we estimate the performance of the model on the validation data set.

  _predictRf <- predict(modelRf, testData)_
  
  _confusionMatrix(testData$classe, predictRf)_
  
  _accuracy <- postResample(predictRf, testData$classe)_
  
  _accuracy_
  
  _oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])_
  
  _oose_
  
So, the estimated accuracy of the model is 99.42% and the estimated out-of-sample error is 0.58%.

<h2> Data Prediction for Test Data </h2> 

Now, we apply the model to the original testing data set downloaded from the data source. We remove the problem_id column first.

  _result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])_
  
  _result_

<h2> Result Visualization </h2> 

Correlation Matrix Visualization
  
  _corrPlot <- cor(trainData[, -length(names(trainData))])_
  
  _corrplot(corrPlot, method="color")_

Decision Tree Visualization
  
  _treeModel <- rpart(classe ~ ., data=trainData, method="class")_
  
  _prp(treeModel) # fast plot_
