Homework3-oscar
================
2023-02-08

``` r
library('splines')        ## for 'bs'
library('dplyr')          ## for 'select', 'filter', and others
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('magrittr')       ## for '%<>%' operator
library('glmnet')         ## for 'glmnet'
```

    ## Loading required package: Matrix

    ## Loaded glmnet 4.1-6

``` r
###  Linear regression examples ###

## load prostate data
prostate <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'))

## split prostate into testing and training subsets
prostate_train <- prostate %>%
  filter(train == TRUE) %>% 
  select(-train)

#summary(prostate_train)

prostate_test <- prostate %>%
  filter(train == FALSE) %>% 
  select(-train)
```

``` r
x <- prostate %>%
  select(-c("train","lcavol","lpsa"))

y <- prostate %>%
  select(-c("train","pgg45","lpsa"))

mat <- cor(x,y)
mat[upper.tri(mat)] <- NA
mat
```

    ##            lcavol    lweight       age         lbph       svi       lcp
    ## lweight 0.2805214         NA        NA           NA        NA        NA
    ## age     0.2249999 0.34796911        NA           NA        NA        NA
    ## lbph    0.0273497 0.44226440 0.3501859           NA        NA        NA
    ## svi     0.5388450 0.15538490 0.1176580 -0.085843238        NA        NA
    ## lcp     0.6753105 0.16453714 0.1276678 -0.006999431 0.6731112        NA
    ## gleason 0.4324171 0.05688209 0.2688916  0.077820447 0.3204122 0.5148301
    ## pgg45   0.4336522 0.10735379 0.2761124  0.078460018 0.4576476 0.6315282
    ##           gleason
    ## lweight        NA
    ## age            NA
    ## lbph           NA
    ## svi            NA
    ## lcp            NA
    ## gleason        NA
    ## pgg45   0.7519045

``` r
## predict lcavol consider all other predictors
## lm fits using L2 loss
fit <- lm(lcavol ~ ., data=prostate_train)

## functions to compute testing/training error w/lm
L2_loss <- function(y, yhat)
  (y-yhat)^2
error <- function(dat, fit, loss=L2_loss)
  mean(loss(dat$lcavol, predict(fit, newdata=dat)))

## train_error 
error(prostate_train, fit)
```

    ## [1] 0.4383709

``` r
## testing error
error(prostate_test, fit)
```

    ## [1] 0.5084068

``` r
## use glmnet to fit ridge
## glmnet fits using penalized L2 loss
## first create an input matrix and output vector
form  <- lcavol ~  lweight + age + lbph + lcp + pgg45 + lpsa + svi + gleason
x_inp <- model.matrix(form, data=prostate_train)
y_out <- prostate_train$lcavol
fit <- glmnet(x=x_inp, y=y_out, alpha = 0, lambda=seq(0.5, 0, -0.05))
#print(fit$beta)

## functions to compute testing/training error with glmnet
error <- function(dat, fit, lam, form, loss=L2_loss) {
  x_inp <- model.matrix(form, data=dat)
  y_out <- dat$lcavol
  y_hat <- predict(fit, newx=x_inp, s=lam)  ## see predict.elnet
  mean(loss(y_out, y_hat))
}

## train_error at lambda=0
error(prostate_train, fit, lam=0, form=form)
```

    ## [1] 0.4383709

``` r
## testing error at lambda=0
error(prostate_test, fit, lam=0, form=form)
```

    ## [1] 0.5083923

``` r
## train_error at lambda=0.03
error(prostate_train, fit, lam=0.05, form=form)
```

    ## [1] 0.4417309

``` r
## testing error at lambda=0.03
error(prostate_test, fit, lam=0.05, form=form)
```

    ## [1] 0.4950521

``` r
## compute training and testing errors as function of lambda
err_train_1 <- sapply(fit$lambda, function(lam) 
  error(prostate_train, fit, lam, form))
err_test_1 <- sapply(fit$lambda, function(lam) 
  error(prostate_test, fit, lam, form))

## plot test/train error
plot(x=range(fit$lambda),
     y=range(c(err_train_1, err_test_1)),
     xlim=rev(range(fit$lambda)),
     type='n',
     xlab=expression(lambda),
     ylab='train/test error')
points(fit$lambda, err_train_1, pch=19, type='b', col='darkblue')
points(fit$lambda, err_test_1, pch=19, type='b', col='darkred')
legend('topright', c('train','test'), lty=1, pch=19,
       col=c('darkblue','darkred'), bty='n')
```

![](Homewrok3-oscar_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
colnames(fit$beta) <- paste('lam =', fit$lambda)
#print(fit$beta %>% as.matrix)
```

``` r
## plot path diagram
plot(x=range(fit$lambda),
     y=range(as.matrix(fit$beta)),
     type='n',
     xlab=expression(lambda),
     ylab='Coefficients')
for(i in 1:nrow(fit$beta)) {
  points(x=fit$lambda, y=fit$beta[i,], pch=19, col='#00000055')
  lines(x=fit$lambda, y=fit$beta[i,], col='#00000055')
}
text(x=0, y=fit$beta[,ncol(fit$beta)], 
     labels=rownames(fit$beta),
     xpd=NA, pos=4, srt=45)
abline(h=0, lty=3, lwd=2)
```

![](Homewrok3-oscar_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->
