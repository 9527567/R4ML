library(h2o)
h2o.init(
    nthreads = 6,
    max_mem_size = "8G"
)
# 此处地址需要输入绝对路径
train_data <- h2o.importFile("/home/jack/workhome/R4ML/mnist_train.csv")
test_data <- h2o.importFile("/home/jack/workhome/R4ML/mnist_test.csv")
y_train <- as.factor(as.matrix(train_data[, 1]))
y_test <- as.factor(as.matrix(test_data[, 1]))
model <- h2o.deeplearning(
    x = 2:785,
    y = 1,
    training_frame = train_data,
    activation = "Tanh",
    hidden = c(100, 1000, 100),
    epochs = 100
)
model

yhat_train <- h2o.predict(model, train_data)$predict
yhat_train <- as.factor(as.matrix(yhat_train))
ytrain <- as.numeric(as.character((y_train)))
ytrain_hat <- as.numeric(as.character(yhat_train))
e <- 0
for (i in 1:60000) {
  if (ytrain[i] <- round(ytrain_hat[i])) {
    e <- e + 1
  }
}

yhat_test <- h2o.predict(model, test_data)$predict
yhat_test <- as.factor(as.matrix(yhat_test))
ytest <- as.numeric(as.character(y_test))
ytest_hat <- as.numeric(as.character(yhat_test))
s <- 0
for (i in 1:10000) {
  if (ytest[i] == round(ytest_hat[i])) {
    s <- s + 1
  }
}
train_acc <- e / 60000
test_acc <- s / 10000

print(paste("train_acc:", train_acc))
print(paste("test_acc:", test_acc))