library(mlr3)
library(tidyverse)
library(mlr3verse)
library(kknn)
library(future)



# 读取训练数据
train_data <- read_csv("mnist_train.csv")
train_data <- as_tibble(train_data)


# python生成的列名
pytemp <- read_lines("a.txt")
# 将label列改名为numbers
names(train_data) <- c("numbers", pytemp)
# 分类目标列必须是factor类型的
train_data <- train_data %>%
    mutate(across(.cols = "numbers", as.factor))
train_data <- as.data.table(train_data)
# 替换原来的数字，将label列更明显的显示
train_data %>%
    mutate(numbers = case_when(
        numbers == 0 ~ "NUM0",
        numbers == 1 ~ "NUM1",
        numbers == 2 ~ "NUM2",
        numbers == 3 ~ "NUM3",
        numbers == 4 ~ "NUM4",
        numbers == 5 ~ "NUM5",
        numbers == 6 ~ "NUM6",
        numbers == 7 ~ "NUM7",
        numbers == 8 ~ "NUM8",
        numbers == 9 ~ "NUM9",
        TRUE ~ "fuck"
    ))

future::plan("multiprocess")
# 创建任务，是一个分类对象
task_train <- TaskClassif$new(
    id = "mnist_nums",
    backend = train_data,
    target = "numbers"
)

print("#########")
# 创建学习者，设定方法和超参数，超参数也可以通过其他方式设置
learner <- lrn("classif.svm", tolerance = 0.001, gamma = 2, kernel = "sigmoid")



# 设置随机数种子，初始的权值
set.seed(123)
# 百分之80作为训练集，其余的为验证集
train <- sample(task_train$nrow, 0.8 * task_train$nrow)
test <- setdiff(seq_len(task_train$nrow), train)

learner$train(task_train, row_ids = train)
# learner$model  # 查看训练好的模型

# 通过已经建立好的模型预测，这里是预测验证集来修正
predictions <- learner$predict(task_train, row_ids = test)
# predictions

# 这里应该是错误的分类，查一下文档
predictions$confusion

# 准确率,acc是accuracy

measure <- msr("classif.acc")
train_acc <- predictions$score(measure)


# 测试集
test_data <- read_csv("mnist_test2.csv")
names(test_data) <- c("numbers", pytemp)
test_data <- as.data.frame(test_data)
test_data <- test_data %>%
    mutate(across(.cols = "numbers", as.factor))
test_data <- as.data.table(test_data)
test_data %>%
    mutate(numbers = case_when(
        numbers == 0 ~ "NUM0",
        numbers == 1 ~ "NUM1",
        numbers == 2 ~ "NUM2",
        numbers == 3 ~ "NUM3",
        numbers == 4 ~ "NUM4",
        numbers == 5 ~ "NUM5",
        numbers == 6 ~ "NUM6",
        numbers == 7 ~ "NUM7",
        numbers == 8 ~ "NUM8",
        numbers == 9 ~ "NUM9",
        TRUE ~ "fuck"
    ))
test_data <- as.data.table(test_data)
task_test <- TaskClassif$new(
    id = "mnist_test",
    backend = test_data,
    target = "numbers"
)
predictions <- learner$predict(task_test)
test_acc <- predictions$score(measure)
print(paste("train_acc:", train_acc))
print(paste("test_acc:", test_acc))
print(paste("测试集和验证集的准确度差值", test_acc - train_acc))