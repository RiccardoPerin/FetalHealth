setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Loading of dataset
data = read.csv('fetal_health.csv', stringsAsFactors = T)

#Check dataset and convert into factors (no missing data)
str(data)
summary(data)

unique(data$severe_decelerations) #Only 2 values
unique(data$histogram_tendency) #Only 3 values
data$severe_decelerations = as.factor(data$severe_decelerations)
data$histogram_tendency = as.factor(data$histogram_tendency)
data$fetal_health = as.factor(data$fetal_health)
levels(data$fetal_health) = c("Normal", "Suspect", "Pathological")

#Target data distribution
table(data$fetal_health) / nrow(data)

#### DECISION TREE ####
library(caret)
library(rpart)
library(rattle)
library(pROC)

#Creation of training and test set
set.seed(123)
idx = createDataPartition(data$fetal_health, p=0.7, list=F)
train_data = data[idx,]
test_data = data[-idx,]

tree0 = rpart(fetal_health~., data=train_data)
fancyRpartPlot(tree0)
plotcp(tree0)
printcp(tree0)

#Need to prune the tree to avoid overfitting
tree.pruned = prune(tree0, cp=0.034)
fancyRpartPlot(tree.pruned)

importance.tree = varImp(tree.pruned)

pred.tree = predict(tree.pruned, test_data, type='class')
confusionMatrix(pred.tree, test_data$fetal_health) #Accuracy of 0.917

#### RANDOM FOREST ####
library(randomForest)
set.seed(123)
tune_grid = expand.grid(.mtry = c(2, 4, 6, 8, 10, 14, 20))
control = trainControl(method = "cv", number = 5)

rf.models = list()
for (trees in c(50, 100, 300, 500)) {
  set.seed(123)
  fit = train(fetal_health ~ ., 
              data = train_data, 
              method = "rf", 
              metric = "Accuracy", 
              tuneGrid = tune_grid, 
              trControl = control,
              ntree = trees)
  
  # Store the model labeled by its tree count
  rf.models[[paste0("Trees_", trees)]] = fit
}
sapply(rf.models, function(x) max(x$results$Accuracy)) #Best model has 100 trees

rf = rf.models[["Trees_100"]]
pred.rf = predict(rf, newdata=test_data)
confusionMatrix(pred.rf, test_data$fetal_health) #Accuracy of 0.943

#### GRADIENT BOOSTING ####
library(gbm)
set.seed(123)
tune_grid = expand.grid(n.trees = c(50, 100, 150, 200),
                        n.minobsinnode = c(5, 10, 15),
                        shrinkage = c(0.01, 0.1),
                        interaction.depth=c(1,2))
gbm.model = train(fetal_health~., 
                  data=train_data, 
                  method='gbm',
                  metric='Accuracy',
                  tuneGrid = tune_grid,
                  trControl = control)

pred.gbm = predict(gbm.model, test_data)
confusionMatrix(pred.gbm, test_data$fetal_health) #Accuracy of 0.948