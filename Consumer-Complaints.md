---
title: "Consumer complaints in the US"
author: "Ali Unlu"
date: "7/30/2021"
output: 
  html_document: 
    keep_md: yes
---



Here, I will use consumer complaints data from the Consumer Complaint Database (CFPB) {link}<https://www.consumerfinance.gov/data-research/consumer-complaints/>. The CFPB is an independent agency of the United States government that promotes transparency and protects consumers by providing information needed to make decisions when choosing financial institutions including banking institutions, lenders, mortgage services, credit unions, securities firms, foreclosure services, and debt collectors. One of the purposes of the agency is to receive and process complaints and questions about consumer financial products and services.   

When a complaint is submitted by a consumer, the CFPB has to determine which category the complaint falls in (e.g. "Mortgage", "Student loan", etc). In this project, I will build a classification algorithm to classify consumer complaints into one of four categories: "Credit card or prepaid card", "Mortgage", "Student loan", or "Vehicle loan or lease". 

# Data 


```r
# packages

library(here)
library(tidymodels)
```


```r
# data
# data

# model building data
df<- read.csv(here("Peer", "data", "data_complaints_train.csv"))
# submission data
q_test <- read.csv(here("Peer", "data", "data_complaints_test.csv")) 

glimpse(df)
```

```
## Rows: 90,975
## Columns: 6
## $ Product                      <chr> "Credit card or prepaid card", "Mortgage"…
## $ Consumer.complaint.narrative <chr> "I initially in writing to Chase Bank in …
## $ Company                      <chr> "JPMORGAN CHASE & CO.", "Ditech Financial…
## $ State                        <chr> "CT", "GA", "IN", "MI", "MI", "FL", "WA",…
## $ ZIP.code                     <chr> "064XX", "None", "463XX", "490XX", "480XX…
## $ Submitted.via                <chr> "Web", "Web", "Web", "Web", "Web", "Web",…
```

```r
# outcome as factor
df$Product <- as.factor(df$Product)

######### data splitting
# initial split
split_df <- rsample::initial_split(data = df, prop = 2/3)

train_df<-rsample::training(split_df)
test_df <-rsample::testing(split_df)
dim(train_df); dim(test_df)
```

```
## [1] 60650     6
```

```
## [1] 30325     6
```

In this type of situation, the narrative of the complaints lead us a direction to build a ML model to classify the complaints inti different groups. We will first create a recipe for the model and then we tokenize the content of the complaint narratives. We followed the path taught on course and create workflow and baking procedures. 


```r
####### recipe

# recipe
recete <-
    recipe(Product ~ Consumer.complaint.narrative, data = train_df)

# tokenization
# install.packages("textrecipes")
library(textrecipes)

recete <- recete %>%
    step_tokenize(Consumer.complaint.narrative) %>% # tokenize the text to words 
    step_tokenfilter(Consumer.complaint.narrative, max_tokens = 1e3) %>% # keep the 1000 most frequent tokens
    step_tfidf(Consumer.complaint.narrative) # to compute tf-idf.

#  preprecess
prep_cc <- prep(recete, verbose = TRUE, retain = TRUE, data=train_df)
```

```
## oper 1 step tokenize [training] 
## oper 2 step tokenfilter [training] 
## oper 3 step tfidf [training] 
## The retained training set is ~ 463.11 Mb  in memory.
```

```r
# names(prep_cc)


baked_train <- bake(prep_cc, new_data = NULL) # we are using the training data.
#dim(baked_train); glimpse(baked_train)

# apply test
baked_test <- recipes::bake(prep_cc, new_data = test_df)
# dim(baked_test); glimpse(baked_test)
```

# Model Building

Among many other models, I used Multinomial regression here to predict the complaint classifications. Since the analysis takes longer time, I used parallel computation to speed the process. 


```r
############# Multinomial regression ##############

mr_model <- multinom_reg(penalty = double(1), mixture = double(1)) %>% 
                 set_engine("glmnet") %>% 
                 translate()


# put this all together into a workflow:
mr_fw <- workflows::workflow() %>%
    workflows::add_recipe(recete) %>%
    workflows::add_model(mr_model)
mr_fw
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: multinom_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 3 Recipe Steps
## 
## • step_tokenize()
## • step_tokenfilter()
## • step_tfidf()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Multinomial Regression Model Specification (classification)
## 
## Main Arguments:
##   penalty = 0
##   mixture = double(1)
## 
## Computational engine: glmnet 
## 
## Model fit template:
## glmnet::glmnet(x = missing_arg(), y = missing_arg(), weights = missing_arg(), 
##     alpha = double(1), family = "multinomial")
```

```r
# do parallel
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## 
## Attaching package: 'foreach'
```

```
## The following objects are masked from 'package:purrr':
## 
##     accumulate, when
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

# model fit
mr_fit <- parsnip::fit(mr_fw, data = train_df)
mr_fit
```

```
## ══ Workflow [trained] ══════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: multinom_reg()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 3 Recipe Steps
## 
## • step_tokenize()
## • step_tokenfilter()
## • step_tfidf()
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## 
## Call:  glmnet::glmnet(x = maybe_matrix(x), y = y, family = "multinomial",      alpha = ~double(1)) 
## 
##       Df  %Dev  Lambda
## 1   1000  0.00 264.700
## 2   1000  0.61 241.100
## 3   1000  0.67 219.700
## 4   1000  0.73 200.200
## 5   1000  0.80 182.400
## 6   1000  0.88 166.200
## 7   1000  0.96 151.400
## 8   1000  1.06 138.000
## 9   1000  1.16 125.700
## 10  1000  1.27 114.600
## 11  1000  1.39 104.400
## 12  1000  1.52  95.110
## 13  1000  1.67  86.660
## 14  1000  1.82  78.960
## 15  1000  2.00  71.950
## 16  1000  2.18  65.560
## 17  1000  2.39  59.730
## 18  1000  2.61  54.430
## 19  1000  2.86  49.590
## 20  1000  3.12  45.190
## 21  1000  3.41  41.170
## 22  1000  3.72  37.510
## 23  1000  4.06  34.180
## 24  1000  4.43  31.140
## 25  1000  4.82  28.380
## 26  1000  5.25  25.860
## 27  1000  5.72  23.560
## 28  1000  6.22  21.470
## 29  1000  6.76  19.560
## 30  1000  7.34  17.820
## 31  1000  7.97  16.240
## 32  1000  8.64  14.800
## 33  1000  9.35  13.480
## 34  1000 10.12  12.280
## 35  1000 10.93  11.190
## 36  1000 11.80  10.200
## 37  1000 12.72   9.292
## 38  1000 13.69   8.467
## 39  1000 14.71   7.715
## 40  1000 15.79   7.029
## 41  1000 16.91   6.405
## 42  1000 18.09   5.836
## 43  1000 19.32   5.317
## 44  1000 20.60   4.845
## 45  1000 21.92   4.415
## 46  1000 23.29   4.022
## 
## ...
## and 54 more lines.
```

```r
# De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```

Now it is time to look out our prediction score. 


```r
########## accuracy 
## predict

pred_product <- bind_cols(
    predict(mr_fit, train_df),
    predict(mr_fit, train_df, type = "prob")
)

yardstick::accuracy(train_df, 
                    truth = Product, estimate = pred_product$.pred_class)
```

```
## # A tibble: 1 × 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy multiclass     0.941
```

The analysis shows that the model produces prety good accuracy results. We have 0.941 accuracy rate. The model could be improved with cross-validation and tuning to strengthen it but for the assignment, I finalized here and apply the model to test data


```r
# Perform prediction
predictSubmission <- predict(mr_fit, q_test , type="class")
predictSubmission
```

```
## # A tibble: 20 × 1
##    .pred_class                
##    <fct>                      
##  1 Student loan               
##  2 Vehicle loan or lease      
##  3 Student loan               
##  4 Mortgage                   
##  5 Credit card or prepaid card
##  6 Credit card or prepaid card
##  7 Credit card or prepaid card
##  8 Credit card or prepaid card
##  9 Student loan               
## 10 Credit card or prepaid card
## 11 Student loan               
## 12 Mortgage                   
## 13 Credit card or prepaid card
## 14 Credit card or prepaid card
## 15 Mortgage                   
## 16 Credit card or prepaid card
## 17 Mortgage                   
## 18 Student loan               
## 19 Mortgage                   
## 20 Mortgage
```


