source("./mlr3_function.R")
options(warn = -1)

# 读取训练集，以及测试集，此处测试集用于观察模型是否过拟合
train_data <- read_Data("mnist_train.csv")
test_data <- read_Data("mnist_test.csv")
# rpart
rpart_result <- learn_rpart(train_data)
print(paste("rpart train Accuracy:", rpart_result[[1]]))

rpart_train_result_acc <- learn_predictions(train_data, rpart_result[[2]])
rpart_test_result_acc <- learn_predictions(test_data, rpart_result[[2]])
print(paste("rpart train Accuracy:", rpart_train_result_acc[[1]]))
print(paste("rpart test Accuracy:", rpart_test_result_acc[[1]]))


# svm
svm_result <- learn_svm(train_data)
print(paste("svm train Accuracy:", svm_result[[1]]))

svm_train_result_acc <- learn_predictions(train_data, svm_result[[2]])
svm_test_result_acc <- learn_predictions(test_data, svm_result[[2]])
print(paste("svm train Accuracy:", svm_train_result_acc[[1]]))
print(paste("svm test Accuracy:", svm_test_result_acc[[1]]))

# kknn
kknn_result <- learn_kknn(train_data)
print(paste("kknn train Accuracy:", kknn_result[[1]]))

kknn_train_result_acc <- learn_predictions(train_data, kknn_result[[2]])
kknn_test_result_acc <- learn_predictions(test_data, kknn_result[[2]])
print(paste("kknn train Accuracy:", kknn_train_result_acc[[1]]))
print(paste("kknn test Accuracy:", kknn_test_result_acc[[1]]))

# naive_bayes
naive_bayes_result <- learn_naive_bayes(train_data)
print(paste("naive_bayes train Accuracy:", naive_bayes_result[[1]]))

naive_bayes_train_result_acc <- learn_predictions(train_data, naive_bayes_result[[2]])
naive_bayes_test_result_acc <- learn_predictions(test_data, naive_bayes_result[[2]])
print(paste("naive_bayes train Accuracy:", naive_bayes_train_result_acc[[1]]))
print(paste("naive_bayes test Accuracy:", naive_bayes_test_result_acc[[1]]))

# ranger
ranger_result <- learn_ranger(train_data)
print(paste("ranger train Accuracy:", ranger_result[[1]]))

ranger_train_result_acc <- learn_predictions(train_data, ranger_result[[2]])
ranger_test_result_acc <- learn_predictions(test_data, ranger_result[[2]])
print(paste("ranger train Accuracy:", ranger_train_result_acc[[1]]))
print(paste("ranger test Accuracy:", ranger_test_result_acc[[1]]))
