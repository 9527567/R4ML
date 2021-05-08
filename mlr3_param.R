library(mlr3verse)
learner <- lrn("classif.svm")
learner$param_set$kernel

