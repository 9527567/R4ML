library(tidyverse)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
read_Data <- function(filename) {
  data <- readr::read_csv(filename)
  pylist <- readr::read_lines("a.txt")
  names(data) <- c("numbers", pylist)
  data <- data %>%
            mutate(dplyr::across(.cols = "numbers", as.factor))
  data <- mlr3::as.data.table(data)
  data %>%
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
  return(data)
}
learn_predictions <- function(data, learner) {
  task_predictions <- mlr3::TaskClassif$new(
            id <- "mnist_predictions",
            backend <- data,
            target <- "numbers"
      )
  predictions <- learner$predict(task_predictions)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(train_acc)
}
# 单个分类树，默认参数准确率63%
learn_rpart <- function(data,
                        cp = 0.01,
                        minsplit = 1,
                        minbucket = 1,
                        maxcompete = 4,
                        maxsurrogate = 5,
                        maxdepth = 30,
                        usesurrogate = 2,
                        surrogatestyle = 0,
                        xval = 10,
                        keep_model = FALSE) {
  task_train <- mlr3::TaskClassif$new(
            id <- "mnist_rpart",
            backend <- data,
            target <- "numbers"
      )
  learner <- mlr3::lrn("classif.rpart",
            cp = cp,
            minsplit = minsplit,
            minbucket = minbucket,
            maxcompete = maxcompete,
            maxsurrogate = maxsurrogate,
            maxdepth = maxdepth,
            usesurrogate = usesurrogate,
            surrogatestyle = surrogatestyle,
            xval = xval,
            keep_model = keep_model
      )
  set.seed(runif(1, min <- -10000, max <- 10000))
  train <- sample(task_train$nrow, 0.8 * task_train$nrow)
  test <- setdiff(seq_len(task_train$nrow), train)
  learner$train(task_train, row_ids <- train)
  predictions <- learner$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  print(paste("before auto tuning,the rpart acc is:", train_acc))
  search_space <- ps(
            cp = p_dbl(lower = 1e-20, upper = 0.1),
            minsplit = p_int(lower = 1, upper = 10)
      )
  measure <- mlr3::msr("classif.ce")
  terminator <- mlr3tuning::trm("evals", n_evals = 10)
  tuner <- mlr3tuning::tnr("grid_search", resolution = 5)
  at <- AutoTuner$new(
            learner = learner,
            resampling = rsmp("holdout"),
            measure = msr("classif.ce"),
            search_space = search_space,
            terminator = terminator,
            tuner = tuner
      )
  at$train(task_train, row_ids <- train)
  predictions <- at$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(list(train_acc, at))
}
# 支持向量机，默认参数准确率11%
learn_svm <- function(data,
                      cachesize = 40,
                      coef0 = 0,
                      cost = 1,
                      cross = 0,
                      decision_values = FALSE,
                      degree = 3,
                      fitted = TRUE,
                      kernel = "sigmoid",
                      gamma = 3,
                      nu = 0.5,
                      scale = TRUE,
                      shrinking = TRUE,
                      tolerance = 0.001,
                      type = "C-classification") {
  task_train <- mlr3::TaskClassif$new(
            id <- "mnist_svm",
            backend <- data,
            target <- "numbers"
      )
  learner <- mlr3::lrn("classif.svm",
            cachesize = cachesize,
            coef0 = coef0,
            cost = cost,
            cross = cross,
            decision.values = decision_values,
            fitted = fitted,
            kernel = kernel,
            gamma = gamma,
            scale = scale,
            shrinking = shrinking,
            tolerance = tolerance,
            type = type
      )
  set.seed(runif(1, min <- -10000, max <- 10000))
  train <- sample(task_train$nrow, 0.8 * task_train$nrow)
  test <- setdiff(seq_len(task_train$nrow), train)
  learner$train(task_train, row_ids <- train)
  predictions <- learner$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  print(paste("before auto tuning,the svm acc is:", train_acc))
  search_space <- ps(
            cost = p_dbl(lower = 0.1, upper = 10),
            kernel = p_fct(levels = c("polynomial", "sigmoid"))
      )
  measure <- mlr3::msr("classif.ce")
  terminator <- mlr3tuning::trm("evals", n_evals = 10)
  tuner <- mlr3tuning::tnr("grid_search", resolution = 5)
  at <- AutoTuner$new(
            learner = learner,
            resampling = rsmp("holdout"),
            measure = msr("classif.ce"),
            search_space = search_space,
            terminator = terminator,
            tuner = tuner
      )
  at$train(task_train, row_ids <- train)
  predictions <- at$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(list(train_acc, at))
}
# k近邻算法，默认参数准确率95%
learn_kknn <- function(data,
                       k = 7,
                       distance = 2,
                       kernel = "optimal",
                       scale = TRUE) {
  task_train <- mlr3::TaskClassif$new(
            id <- "mnist_kknn",
            backend <- data,
            target <- "numbers"
      )
  learner <- mlr3::lrn("classif.kknn",
            k = k,
            distance = distance,
            kernel = kernel,
            scale = scale
      )
  set.seed(runif(1, min <- -10000, max <- 10000))
  train <- sample(task_train$nrow, 0.8 * task_train$nrow)
  test <- setdiff(seq_len(task_train$nrow), train)
  learner$train(task_train, row_ids <- train)
  predictions <- learner$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  print(paste("before auto tuning,the kknn acc is:", train_acc))
  search_space <- ps(
            k = p_int(lower = 1, upper = 100),
            distance = p_int(lower = 0, upper = 50)
      )
  measure <- mlr3::msr("classif.ce")
  terminator <- mlr3tuning::trm("evals", n_evals = 10)
  tuner <- mlr3tuning::tnr("grid_search", resolution = 5)
  at <- AutoTuner$new(
            learner = learner,
            resampling = rsmp("holdout"),
            measure = msr("classif.ce"),
            search_space = search_space,
            terminator = terminator,
            tuner = tuner
      )
  at$train(task_train, row_ids <- train)
  predictions <- at$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(list(train_acc, at))
}
# 朴素贝叶斯
learn_naive_bayes <- function(data,
                              laplace = 0,
                              threshold = 0.001,
                              eps = 0) {
  task_train <- mlr3::TaskClassif$new(
            id <- "mnist_naive_bayes",
            backend <- data,
            target <- "numbers"
      )
  learner <- mlr3::lrn("classif.naive_bayes",
            laplace = laplace,
            threshold = threshold,
            eps = eps
      )
  set.seed(runif(1, min <- -10000, max <- 10000))
  train <- sample(task_train$nrow, 0.8 * task_train$nrow)
  test <- setdiff(seq_len(task_train$nrow), train)
  learner$train(task_train, row_ids <- train)
  predictions <- learner$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  print(paste("before auto tuning,the naive_bayes acc is:", train_acc))
  search_space <- ps(
            threshold = p_dbl(lower = 1e-10, upper = 1.0)
      )
  measure <- mlr3::msr("classif.ce")
  terminator <- mlr3tuning::trm("evals", n_evals = 10)
  tuner <- mlr3tuning::tnr("grid_search", resolution = 5)
  at <- AutoTuner$new(
            learner = learner,
            resampling = rsmp("holdout"),
            measure = msr("classif.ce"),
            search_space = search_space,
            terminator = terminator,
            tuner = tuner
      )
  at$train(task_train, row_ids <- train)
  predictions <- at$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(list(train_acc, at))
}


# 随机森林,参数有很多，挑选数值型参数调整，写入函数形参,很有意思，准确率96%
learn_ranger <- function(data,
                         alpha = 0.5,
                         num.trees = 500,
                         splitrule = "extratrees",
                         num.threads = 1,
                         num.random.splits = 1,
                         min.node.size = 1) {
  task_train <- mlr3::TaskClassif$new(
            id <- "mnist_ranger",
            backend <- data,
            target <- "numbers"
      )
  learner <- mlr3::lrn("classif.ranger",
            alpha = alpha,
            num.trees = num.trees,
            splitrule = splitrule,
            num.threads = num.threads,
            num.random.splits = num.random.splits,
            min.node.size = min.node.size
      )
  set.seed(runif(1, min <- -10000, max <- 10000))
  train <- sample(task_train$nrow, 0.8 * task_train$nrow)
  test <- setdiff(seq_len(task_train$nrow), train)
  learner$train(task_train, row_ids <- train)
  predictions <- learner$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  print(paste("before auto tuning,the ranger acc is:", train_acc))
  search_space <- ps(
            alpha = p_dbl(lower = 0.01, upper = 10),
            num.trees = p_int(lower = 100, upper = 5000)
      )
  measure <- mlr3::msr("classif.ce")
  terminator <- mlr3tuning::trm("evals", n_evals = 10)
  tuner <- mlr3tuning::tnr("grid_search", resolution = 5)
  at <- AutoTuner$new(
            learner = learner,
            resampling = rsmp("holdout"),
            measure = msr("classif.ce"),
            search_space = search_space,
            terminator = terminator,
            tuner = tuner
      )
  at$train(task_train, row_ids <- train)
  predictions <- at$predict(task_train, row_ids <- test)
  measure <- mlr3::msr("classif.acc")
  train_acc <- predictions$score(measure)
  return(list(train_acc, at))
}
# 惩罚逻辑回归,40个参数
learn_cv_glmnet <- function(data) {

}
# 线性判别分析,参数较少，可能参数之间相互依赖
learn_lda <- function(data) {

}
# 逻辑回归,参数适中
learn_log_reg <- function(data) {

}
# 梯度提升，参数非常多
learn_xgboost <- function(data) {

}
# 二次判别分析，参数很少，相互之间有依赖
learn_qda <- function(data) {

}
# 多项式对数模型,参数适中
learn_multinom <- function(data) {

}