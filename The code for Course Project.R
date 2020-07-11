#Download the file from link in the instruction
data1 <- "training.csv"
data2 <- "testing.csv"

if(!file.exists(data1)){
        fileUrl<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        download.file(fileUrl,data1,method = "curl")
}

if(!file.exists(data2)){
        fileUrl<-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        download.file(fileUrl,data2,method = "curl")
}

# Load the data
train_data <- read.csv(data1, strip.white = TRUE, na.strings = c("NA",""))
test_data <- read.csv(data2, strip.white = TRUE, na.strings = c("NA",""))

# Check the data
dim(train_data)
str(train_data)

dim(test_data)
str(test_data)

# Data pre-processing
install.packages("rattle")
library(rattle)
install.packages("randomForest")
library(randomForest)
library(caret)
library(rpart)

# Separate the dataset
training_partition <- createDataPartition(train_data$classe,p=0.75,list = FALSE)
train_subset <- train_data[training_partition,]
test_subset <- train_data[-training_partition,]

# pre-process the subset
low_var <- nearZeroVar(train_subset)
train_subset <- train_subset[,-low_var]
test_subset <- test_subset[,-low_var]

var_na <- sapply(train_subset,function(x){
                                mean(is.na(x))>0.95})
train_subset <- train_subset[,var_na==FALSE]
test_subset <- test_subset[,var_na==FALSE]

train_subset <- train_subset[ , -(1:5)]
test_subset  <- test_subset [ , -(1:5)]

dim(train_subset)
dim(test_subset)

# Apply the correlation analysis for the dataset
install.packages("corrplot")
library(corrplot)
library(lattice)
library(ggplot2)
library(rattle)
library(rpart.plot)
correlation <- cor(train_subset[,-54])
corrplot(correlation, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

# Construct the model
# Plot the decision tree
set.seed(1000)
decision_tree <- rpart(classe~.,data=train_subset,method="class")
fancyRpartPlot(decision_tree)

# Predictions from the decision tree model
prediction_decision_tree <- predict(decision_tree,newdata = test_subset,
                                    type = "class")
conf_decision_tree <- confusionMatrix(prediction_decision_tree,as.factor(test_subset$classe))
conf_decision_tree

plot(conf_decision_tree$table, col = conf_decision_tree$byClass, 
     main = paste("Decision Tree Model: Predictive Accuracy =",
                  round(conf_decision_tree$overall['Accuracy'], 4)))

# Generalized Boosted Model
set.seed(1000)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fit_GBM  <- train(classe ~ ., data = train_subset, method = "gbm",
                  trControl = control_GBM, verbose = FALSE)
fit_GBM$finalModel

predict_GBM <- predict(fit_GBM, newdata=test_subset)
conf_GBM <- confusionMatrix(predict_GBM, as.factor(test_subset$classe))
conf_GBM

plot(conf_GBM$table, col = conf_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(conf_GBM$overall['Accuracy'], 4)))

# Random forest model
set.seed(1000)
control_RFM <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
fit_RFM  <- train(classe ~ ., data = train_subset, method = "rf",
                 trControl = control_RFM, verbose = FALSE)
fit_RFM$finalModel

predict_RFM <- predict(fit_RFM, newdata = test_subset)
conf_RFM <- confusionMatrix(predict_RFM, as.factor(test_subset$classe))
conf_RFM

plot(conf_RFM$table, col = conf_RFM$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(conf_RFM$overall['Accuracy'], 4)))

predict_test <- predict(fit_RFM, newdata=test_data)
predict_test


