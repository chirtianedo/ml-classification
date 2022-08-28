
### This project was created to compare the accuracies of two different models 
# in predicting the price of a car. 
# It was a competition  posted on kaggle about a year ago (I think). 
# So without futher ado lets dive into it...


# 1. We start with loading our libraries/packages.
# I generally love using the "pacman" package as it allows to 
# load two or more packages together as well as installing packages directly 
# if they were absent from our tools

library(pacman)

p_load(gridExtra, tidyverse, rpart.plot, caret, labelVector, standardize)


   

# 
# 2. I think we're good to load the data now. There exists many forms of this data. The original dataset was only available to about 8 teams on kaggle.
# That set had about 77 columns/variables and their accurate descriptions. 
# 
# My dataset was downloaded from just searching Toyota.csv dataset online. I really couldnt find the link since my PC shutdown completely, but I included the csv file in the folder. Help yourself.
# 
# Description:
# - Price: The current price of the automobile
# - Age: How long in months since the release date of the car
# - KM:Accumulated kilometers on odometer
# - HP: Obviously, horsepower
# - MetColor: Mettalic colour Yes = 1, No= 0
# - CC: cylinder Volume
# - Doors: Number of doors
# - Weight: Weight in kilograms
# - Fuel_Diesel: Fuel or Diesel type 
   {r Toyota dataset}
df <- read.csv("ToyotaCorolla.csv")
head(df)
summary(df)
   
# We can tell the behavior and type of each variable from the summary and the calling the head function. Those with the min and max values show two different levels as compared to the strictly numerical values.
# 
# 
# I checked the distribution of Price in the dataset and as expected, higher car prices had lower count values, while the peak count/distribution value hovered around the $10,000 mark.
  

ggplot(data=df, aes(x = Price)) + 
  geom_histogram(colour = "grey", fill = "black") +
  ylim (0, 300) +
  ggtitle("Original Price Distribution") +
  labs(x = "Price", y ="Distribution Frequency") + theme_bw()

   
# 
# There are lot of visualization techniques to give an overview of the 
# data structure but I ended at the simple visual above. 
# I went ahead to split the data set before any transformation 
# process to avoid data leakage. I usually do this before visualizng, 
# but it still works fine, as long as we are not cheating

  #  - So I split the data by 80% into a train and a test variable..
  #  - I standardize the data the dataset (without the dependent variable)
  #  - Then I create a copy of the dataset before standardizing
  
trainRows <- createDataPartition(y = df$Price, p = 0.85, list = FALSE)#the createDataPartition function helps to randomly separate rows into 85% and !5% for the training and testing data respectively

train_set <- df[trainRows,] #subset of df in trainrows
test_set <- df[-trainRows,] #subset of df outside trainrows

#As stated earlier
train_set_stand <- train_set
test_set_stand <- test_set
   

# The next step is to apply standardization measures to the training and 
# testing datasets.
  

train_set_stand[,2:10] <- apply(train_set_stand[,2:10], MARGIN = 2, FUN = scale)#ignoring column 1 because it represent the predicting variable
test_set_stand[,2:10] <- apply(test_set_stand[,2:10], MARGIN = 2, FUN = scale)

   

# I move on with running the algorithm function on the training dataset.
# When we run the algorithm function on the standardized train sets... 
# with method = knn,the algorithm will look at the outcome variable and 
# deduce if its a classification or a numerical prediction task.
# When it runs, we get predictions from the k-NN model and use ggplot to 
# create a histogram of the distribution of the true price value 
# using the testing data, and compare to the training data, 
# to see how the model is performing.

  
knn_model <- train(Price~., train_set_stand, method = "knn")
knn_model 
   

# After letting the model decide the best neighbouring value, 
# we can confirm or observe the interpretation by plotting the 
# root mean square error of each value selected from the grid points RMSE
  

grid = expand.grid(k = c(5,7,9,13,15)) #gives a lot of point with uniform intervals for knn to work with

knn2 <- train(data= train_set_stand, method = "knn", Price~., 
              trControl = trainControl(search="grid"), tuneGrid=grid) 
knn2 #The final model selected for the model was k=9.
   

  
plot(knn2, ylab = "RMSE")
   

# We can now get our predictions from the KNN model and compare 
# with true values in test data
  
knnPred <- predict(knn_model, test_set_stand)

#Get a histogram of the predicted prices
h_pred_knn<- ggplot(data= test_set_stand, aes(x = knnPred)) + 
  geom_histogram(colour = "lightblue", fill = "darkblue") +
  xlim (0, 33000) + 
  ggtitle("KNN, Distribution of Predicted Price") +
  labs(x = "Predicted Price") + theme_linedraw() + scale_y_continuous(breaks = seq(0, 50, 10))

#histo of true prices test data
price_dist<- ggplot(data=test_set, aes(x = Price)) + 
  geom_histogram(colour = "grey", fill = "black") +
  xlim (0, 33000) +
  ggtitle("Original Price Distribution") +
    labs(x = "True Price") + theme_linedraw() + scale_y_continuous(breaks = seq(0, 50, 10))

#grid.arrange function helps to compare plots together
grid.arrange(price_dist, h_pred_knn, nrow=1)
   

# From our plots we can tell that our KNN is performing quite well 
# even if its overpredicting a lot of values, the shape of the 
# distribution behaves or looks the same.
# 
# We could also visualize the perfromance through computation and
# plotting the error metrics 
  
#This computes the prediction error
knn_error <-knnPred - test_set_stand$Price

#histogram of the prediction error
h_error_knn <- ggplot(data= test_set_stand, aes(x = knn_error)) + 
  geom_histogram(colour = "lightblue", fill = "blue") +
  ylim (0, 150) + 
  ggtitle("KNN, Distribution of Prediction Error") +
  labs(x = "Prediction Error")



# Plots prediction error agaisnt actual price
p_error_knn<- ggplot(data = test_set_stand, aes(x=Price, y=knn_error)) +
  geom_point(size=2, color = "blue") +
  xlim (0, 30000) +
  ggtitle("KNN, Prediction Error vs Actual Price") +
  labs(x = "True Price", y = "KNN Prediction Error")

grid.arrange(h_error_knn, p_error_knn)
   


# Lastly I computed the error metrics of the model used and put them together.
  

#Mean error: take the mean of the prediction error
knnME <- mean(knn_error)
knnME

#MAE - Mean absolute error: use th MAE function
knnMAE <- MAE(pred = knnPred, obs = test_set_stand$Price)
knnMAE

#MAPE - Mean absolute % error
#First, take the absolute value of the ratio between error and actual value
knn_abs_relative_error <-abs(knn_error/test_set_stand$Price)

#Then take the average and * by 100
knnMAPE <- mean(knn_abs_relative_error)*100
knnMAPE

#RMSE - Root mean square error: use the RMSE function
knnRMSE<- RMSE(pred = knnPred, obs = test_set_stand$Price)
knnRMSE

#Put together the error measures
res_knn<- c(knnME, knnMAE, knnMAPE, knnRMSE)
names(res_knn) <-c("ME", "MAE", "MAPE", "RMSE")
res_knn <- set_label(res_knn, "KNN")
res_knn



   



##REGRESSION TREE

# So we've seen the performance of the KNN on a dataset, 
# I want to compare the performance of a regression tree against what we ran before
# 
# Since we have the split test and train dataset, 
# we can just apply the r.part() function to the data.
# We do not need to standardized since the regression model does 
# not involve a distance measuring property.


  
#Train regression tree; use the non standardized data

rtree <- train(Price~., data = train_set, method = "rpart")

rpart.plot(rtree$finalModel, digits=-3)
   


# Get predictions from the model I ran earlier plot a histogram and 
# compare like we did with KNN model to examine performance
  

treePred <- predict(rtree, test_set)

#Get a histogram of the predicted prices
h_pred_tree<- ggplot(data= test_set, aes(x = treePred)) + 
  geom_histogram(colour = "red", fill = "darkred") +
  xlim (0,30000) + 
  ylim (0, 150) + 
  ggtitle("Tree, Distribution of Predictions") +
  labs(x = "Predictions")

#compare to the actual price distribution we created above
grid.arrange(price_dist,h_pred_tree, nrow=1)

   


  
#Error Metrics
#Prediction error
tree_error <-treePred - test_set$Price

#Visualize prediction error
#Histogram of the distribution of prediction error
h_error_tree<- ggplot(data= test_set, aes(x = tree_error)) + 
  geom_histogram(colour = "darkred", fill = "red") +
  xlim (-6000, 6000) + 
  ylim (0, 150) + 
  ggtitle("Tree, Distribution of Prediction Error") +
  labs(x = "Prediction Error")

#Plot prediction error vs actual price
p_error_tree<- ggplot(data = test_set, aes(x=Price, y=tree_error)) +
  geom_point(size=2, color = "red") +
  ylim (-6000, 8000) +
  xlim (0, 30000) +
  ggtitle("Tree, Prediction Error vs Actual Price") +
  labs(x = "Actual Price", y = "Tree Prediction Error")

grid.arrange(h_error_tree, p_error_tree)
 
# For a fairly good prediction performance, 
# we'd like for as many values or observation to be close to zero. 
# At the end of these, I put together side by side the computation of 
# error metrics of each model I performed on the dataset.

#Mean error
ME_tree <- mean(tree_error)
#MAE
treeMAE <- MAE(pred = treePred, obs = test_set$Price)
#MAPE
tree_abs_relative_error <-abs(tree_error/test_set$Price)
treeMAPE <- mean(tree_abs_relative_error)*100
#RMSE
treeRMSE <- RMSE(pred = treePred, obs = test_set$Price)
#Put together the performance measures
res_tree<- c(ME_tree, treeMAE, treeMAPE, treeRMSE)

names(res_tree) <-c("ME", "MAE", "MAPE", "RMSE")

res_tree <- set_label(res_tree, "Regression Tree")
res_tree

# Then I compared the distribution of the original price with the 
# distribution of predicted prices in both models

grid.arrange(price_dist, h_pred_knn,h_pred_tree, nrow = 1)

grid.arrange(p_error_knn,p_error_tree, nrow = 2)

grid.arrange(h_error_knn,h_error_tree, nrow = 2)

# On the final note, I compare the error metric values 
# What do you think? which model performs better?

res_knn
res_tree
   

