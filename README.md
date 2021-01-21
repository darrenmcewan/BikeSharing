# BikeSharing
Creating a model to predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

The purpose of this project was to predict the number of bikes rented given weather, and day time variables. 
model.R is my main R script. It has all of the libraries that I imported, feature engineering, and trained xgbTree model. test.csv and train.csv contain the data necessary to complete the project. 

Registered and casual users were not included in our test set and were not part of what we were trying to predict. I removed those two + "feels like" atemp variable because it was highly correlated with our temp variable. I also changed count to log(count) because the data was skewed and our target was RMSLE.

To generate the best predictions, I used xgboost with hyperparameters:
nrounds=200
max_depth = 15
colsample_bytree = 0.8
eta = 0.1
gamma=0
min_child_weight = 1
subsample = 1
