source("./function.R")
options(warn = -1)

# 读取训练集，以及测试集，此处测试集用于观察模型是否过拟合
train_data <- read_Data("mnist_train2.csv")
test_data <- read_Data("mnist_test2.csv")

rpart_result <- learn_kknn(train_data, k = 10)
print(paste("rpart train Accuracy:", rpart_result[[1]]))

rpart_result_acc <- learn_predictions(test_data, rpart_result[[2]])
print(paste("rpart test Accuracy:", rpart_result_acc[[1]]))
# 通过比较训练集准确率与测试集准确率差值观察是否过拟合