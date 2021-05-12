library(mlr3verse)
learner <- lrn("classif.svm")
learner$param_set
learner <- lrn("classif.rpart")
learner$param_set

