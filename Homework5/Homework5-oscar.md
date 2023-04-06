Homework5-oscar
================
2023-04-02

``` r
library(gbm)
```

    ## Loaded gbm 2.1.8.1

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
vowel.train = read.csv('https://hastie.su.domains/ElemStatLearn/datasets/vowel.train')
vowel.test = read.csv('https://hastie.su.domains/ElemStatLearn/datasets/vowel.test')
vowel.train <- vowel.train[, -1]
vowel.test <- vowel.test[, -1]

y_train <- as.numeric(vowel.train[,1])
X_train <- as.matrix(vowel.train[,-1])

y_test <- as.numeric(vowel.test[,1])
X_test <- as.matrix(vowel.test[,-1])
```

``` r
param_grid <- expand.grid(n.trees = seq(from = 50, to = 1000, by = 50), interaction.depth = 1, shrinkage = 0.1, n.minobsinnode = 10)

train_control <- trainControl(method = "cv", number = 5, verboseIter = FALSE, allowParallel = TRUE)

model <- train(x = X_train,y = y_train, method = "gbm",trControl = train_control, tuneGrid = param_grid, verbose = FALSE)

print(model)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 528 samples
    ##  10 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 424, 422, 422, 422, 422 
    ## Resampling results across tuning parameters:
    ## 
    ##   n.trees  RMSE      Rsquared   MAE     
    ##     50     1.981218  0.6270816  1.569012
    ##    100     1.820940  0.6737278  1.429808
    ##    150     1.745498  0.6983097  1.370013
    ##    200     1.709063  0.7093603  1.331524
    ##    250     1.675182  0.7200146  1.306395
    ##    300     1.661618  0.7239265  1.298552
    ##    350     1.648422  0.7281354  1.290175
    ##    400     1.645025  0.7297587  1.289092
    ##    450     1.640181  0.7313957  1.280576
    ##    500     1.631044  0.7343255  1.274551
    ##    550     1.629815  0.7348961  1.268967
    ##    600     1.631741  0.7337497  1.265037
    ##    650     1.627635  0.7352135  1.267175
    ##    700     1.631687  0.7342965  1.264527
    ##    750     1.621669  0.7372444  1.255668
    ##    800     1.614038  0.7396855  1.251045
    ##    850     1.616908  0.7388554  1.254092
    ##    900     1.621762  0.7375196  1.258005
    ##    950     1.616734  0.7389188  1.255557
    ##   1000     1.618882  0.7387560  1.256574
    ## 
    ## Tuning parameter 'interaction.depth' was held constant at a value of 1
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 800, interaction.depth =
    ##  1, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
model <- gbm.fit(X_train, y_train, distribution = "multinomial", n.trees = 300, interaction.depth = 1,shrinkage = 0.1,n.minobsinnode = 10,verbose = FALSE)

test_pred <- predict(model, newdata = X_test, n.trees = 300, type="response")

test_pred_class <- apply(test_pred, 1, which.max)

misclassification_rate <- mean(test_pred_class != y_test)

print(misclassification_rate)
```

    ## [1] 0.534632
