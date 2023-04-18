Homework6-oscar
================
2023-04-08

``` r
library(keras)
zip.test = data.table::fread("https://hastie.su.domains/ElemStatLearn/datasets/zip.test.gz")
zip.train = data.table::fread("https://hastie.su.domains/ElemStatLearn/datasets/zip.train.gz")
```

``` r
# Feature scale RGB values in test and train inputs  
x_train <- as.matrix(zip.train[,-1])
x_test <- as.matrix(zip.test[,-1])
y_train <- zip.train$V1
y_test <- zip.test$V1
```

``` r
x_train <- array(x_train, dim = c(dim(x_train)[1], 16, 16, 1))
x_test <- array(x_test, dim = c(dim(x_test)[1], 16, 16, 1))
```

``` r
# Initialize sequential model
model <- keras_model_sequential()

model <- model %>%
  # Start with hidden 2D convolutional layer being fed 16x16 pixel images
  layer_conv_2d(
    filter = 4, kernel_size = c(3,3), 
    input_shape = c(16, 16, 1),
    activation = 'relu'
  ) %>%

  # Second hidden layer
  layer_conv_2d(filter = 8, kernel_size = c(3,3),
    activation = 'relu') %>%

  # Use max pooling
  #layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(
    filter = 16, kernel_size = c(3,3), 
    activation = 'relu'
  ) %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3),
    activation = 'relu') %>%

  # Use max pooling
  #layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(32) %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(10, activation = 'softmax')

summary(model)
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ##  Layer (type)                       Output Shape                    Param #     
    ## ================================================================================
    ##  conv2d_3 (Conv2D)                  (None, 14, 14, 4)               40          
    ##  conv2d_2 (Conv2D)                  (None, 12, 12, 8)               296         
    ##  conv2d_1 (Conv2D)                  (None, 10, 10, 16)              1168        
    ##  conv2d (Conv2D)                    (None, 8, 8, 32)                4640        
    ##  flatten (Flatten)                  (None, 2048)                    0           
    ##  dense_1 (Dense)                    (None, 32)                      65568       
    ##  dense (Dense)                      (None, 10)                      330         
    ## ================================================================================
    ## Total params: 72,042
    ## Trainable params: 72,042
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

``` r
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Training ----------------------------------------------------------------
history <- model %>% fit(
  x_train, y_train, epochs = 15, verbose = 2
)

model %>% evaluate(x_test, y_test)
```

    ##      loss  accuracy 
    ## 0.3744533 0.9427006

``` r
plot(history$metrics$accuracy, type = "l", col = "blue", xlab = "Epoch", ylab = "Accuracy", ylim = c(0,1))
legend("bottomright", legend = c("Training"), col = c("blue"), lty = 1)
```

![](Homework6-oscar_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
