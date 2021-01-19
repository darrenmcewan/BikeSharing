library(dplyr)
library(lubridate)
library(chron)
library(MLmetrics)
library(caret)
library(randomForest)
library(vroom)
library(DataExplorer)

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

complete <- complete %>% select(-atemp)

## Dummy variable encoding - one-hot encoding
#dummyVars(count~season, data =complete) %>%  predict(complete) %>% as.data.frame() %>% bind_cols(complete %>% select(-season))



train <- complete %>% filter(!is.na(count))
test <- complete %>% filter(is.na(count))

custom_summary = function(data, lev = NULL, model = NULL) {
  library(Metrics)
  out = rmsle(data[, "obs"], data[, "pred"])
  names(out) = c("rmsle")
  out
}

control = trainControl(method = "cv",  
                       number = 10,     
                       summaryFunction = custom_summary)



bike.model <- train(form=count~.-datetime, 
                    data=train,
                    method="ranger"
                    metric="RMSLE",
                    trControl=control, )

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
