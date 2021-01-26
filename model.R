library(tidyverse)
library(lubridate)
library(MLmetrics)
library(caret)



#Import and clean data
train <- read.csv("train.csv")
train <- train %>% select(-casual,-registered)

test <- read.csv("test.csv")
test$count <- NA

train$season <- as.factor(train$season)


#Exploratory Data
ggplot(data=train, aes(x=datetime, y=count, color=season))+geom_point()
ggplot(data=train, aes(x=datetime, y=count, color=month(datetime)))+geom_point()
plot_missing(complete)
plot_correlation(data = complete, type = "continuous", cor_args = "pairwise.complete.obs")
ggplot(data=train, aes(x=hour, y=count, color=day))+geom_point()



complete <- rbind(train,test)
complete$datetime1 <- as.POSIXct(complete$datetime, tz="EST", "%Y-%m-%d %H:%M:%S")
complete$date  = date(complete$datetime1)
complete$day  <- wday(complete$datetime1, label=TRUE)
complete$hour <-  hour(complete$datetime1)
complete$rush_hour <- ifelse((complete$hour == 7 | complete$hour <= 8 & complete$workingday == 1)| (complete$hour >= 17 & complete$hour <= 20 & complete$workingday == 1), TRUE, FALSE)
complete$month <- as.factor(month(complete$datetime1))
complete$year <- as.factor(year(complete$datetime1))
complete$wkendRide <- ifelse((complete$hour >= 10 & complete$hour <= 16) & complete$workingday == 0, TRUE, FALSE)
complete$hour <- as.factor(complete$hour)
complete <- complete %>% select(-date, -datetime1)

#Make sure all variables are factors that need to be
complete$season<- as.factor(complete$season)
complete$holiday<- as.factor(complete$holiday)
complete$workingday<- as.factor(complete$workingday)
complete$weather<- as.factor(complete$weather)


#Highly correlated with temp, so we can drop it
complete <- complete %>% select(-atemp)





#Change to log because the metric for the competition is rmsle and also the distribution of counts is skewed.
complete$count = log1p(complete$count)


#Split test/train
train <- complete %>% filter(!is.na(count))
test <- complete %>% filter(is.na(count))


#Grid space to search for the best hyperparameters
xgbGrid <- expand.grid(nrounds = c(100,200),  
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)


#Specify cross-validation method and number of folds. Also enable parallel computation

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)



#Train the model
gbm <- train(count~.-datetime,
             data = train, 
             trControl = xgb_trcontrol,
             tuneGrid = xgbGrid,
             method = "xgbTree",
             verbose = FALSE,
             type="response")

gbm$bestTune
#   nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
#     200        15   0.1     0              0.8                1         1


#Gather predictions
preds <- predict(gbm, newdata = test)

#Change preds back from log
preds <- expm1(preds)

preds.frame <- data.frame(datetime = test$datetime, count = preds)


write.csv(preds.frame, "xgboost_log1.csv", row.names = FALSE)
