library(dplyr)
library(lubridate)
library(chron)
library(MLmetrics)
library(caret)
library(readr)
library(randomForest)

#Import and clean data
train <- read.csv("train.csv")
train <- train %>% select(-casual,-registered)

test <- read.csv("test.csv")
test$count <- NA

complete <- rbind(train,test)
complete$datetime1 <- as.POSIXct(complete$datetime, tz="EST", "%Y-%m-%d %H:%M:%S")
complete$date  = date(complete$datetime1)
complete$day  <- wday(complete$datetime1, label=TRUE)
complete$hour <-  hour(complete$datetime1)
complete$rush_hour <- ifelse(complete$hour >= 7 & complete$hour <= 10 & complete$workingday == 1| complete$hour >= 4 & complete$hour <= 6 & complete$workingday == 1, TRUE, FALSE)

complete <- complete %>% select(-date, -datetime1)
complete$season<- as.factor(complete$season)
complete$holiday<- as.factor(complete$holiday)
complete$workingday<- as.factor(complete$workingday)
complete$weather<- as.factor(complete$weather)
complete$hour<- as.factor(complete$hour)


train <- complete %>% filter(!is.na(count))
test <- complete %>% filter(is.na(count))

gbm <- train(count~.-datetime,
             data = train, 
             method = "gbm",
             verbose = FALSE,
             type="response"
             
)


preds <- predict(gbm, newdata = test)
preds.frame <- data.frame(datetime = test$datetime, count = preds)
preds.frame$count<- ifelse(preds.frame$count < 0,0,preds.frame$count)


write_csv(preds.frame, "bikeSharegbm.csv")
