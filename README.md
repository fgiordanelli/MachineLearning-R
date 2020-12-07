# MachineLearning-R

---
title: "Machine Learning"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(dplyr)
library(mltools)
library(data.table)
library(caret)
library(RANN)
library(knitr)
library(kableExtra)
library(purrr)
library(randomForest)
library(h2o)
library(gbm)

orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv', stringsAsFactors = TRUE)



head(orange[, 1:10])

set.seed(100)

trainRowNumbers <- createDataPartition(orange$Purchase, p=0.8, list=FALSE)
trainData <- orange[trainRowNumbers,]
testData <- orange[-trainRowNumbers,]

# Store X and Y for later use.
x = trainData[, 2:18]
y = trainData$Purchase

```


## Treino e Teste {.tabset}

### Treino
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

head(trainData) %>% kable() %>% kable_styling() %>% print()

```

### Teste

```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

head(testData) %>% kable() %>% kable_styling() %>% print()

```


```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# Store X and Y for later use.
x = trainData[, 2:18]
y = trainData$Purchase

```


## Valores Vazios {.tabset}

### Antes
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

valor_vazio <- trainData[!complete.cases(trainData), ] 

valor_vazio %>% head() %>% kable() %>% kable_styling() %>% print()

```


### Depois
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

preProcess_missingdata_model <- preProcess(trainData, method='knnImpute')

preProcess_missingdata_model

trainData <- predict(preProcess_missingdata_model, newdata = trainData)

trainData %>% head() %>% kable() %>% kable_styling() %>% print()

```




## Encoding {.tabset}

### Antes
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

trainData %>% select_if(Negate(is.numeric)) %>% head() %>% kable() %>% kable_styling() %>% print()

```


### Ordinal Encoding
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

trainData %>% select_if(Negate(is.numeric)) %>%  sapply(as.integer) %>% head() %>% kable() %>% kable_styling() %>% print()

```



### One-Hot
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

colunas <- colnames(trainData %>% select_if(Negate(is.numeric)))
dmy <- dummyVars(" ~ .", data = trainData)


data.frame(predict(dmy, newdata = trainData)) %>% select(contains(paste0(colunas,"."))) %>% head() %>% kable() %>% kable_styling() %>% print()


# names <- trainData %>% select_if(Negate(is.numeric)) %>% colnames() %>% paste0("_")
# 
# trainData <- as.data.table(trainData)
# trainData <- one_hot(trainData)
# 
# trainData %>% select(contains(names)) %>% head() %>% kable() %>% kable_styling() %>% print()

```


### Dummy 
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```


### Effect
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### Binary
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```


### BaseN
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```


### Hash
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```


### Target
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}



```


## Transformações dos dados {.tabset}

### range
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# preProcess_range_model <- preProcess(trainData, method='range')
# 
# trainData <- predict(preProcess_range_model, newdata = trainData)
# 
# trainData %>% select(contains(names)) %>% head() %>% kable() %>% kable_styling() %>% print()

```


### center
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### scale
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### BoxCox
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```


### YeoJohnson
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### expoTrans
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### pca
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### ica
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```

### spatialSign
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

```




## Importância das variáveis {.tabset}

### BoxPlot
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# trainData$Purchase <- y
# 
# featurePlot(x = trainData[, 1:18], 
#             y = trainData$Purchase, 
#             plot = "box",
#             strip=strip.custom(par.strip.text=list(cex=.7)),
#             scales = list(x = list(relation="free"), 
#                           y = list(relation="free")))

```

### Densidade

```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# featurePlot(x = trainData[, 1:18], 
#             y = trainData$Purchase, 
#             plot = "density",
#             strip=strip.custom(par.strip.text=list(cex=.7)),
#             scales = list(x = list(relation="free"), 
#                           y = list(relation="free")))

```



## Seleção de feature {.tabset}

### Correlação
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# correlationMatrix <- cor(trainData[,1:18])
# 
# correlationMatrix %>% kable() %>% kable_styling() %>% print()

# olhar essa função :  highlyCorrelated 

```


### Rank pela importância
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# varimp_mars <- varImp(trainData)
# plot(varimp_mars, main="Variable Importance with MARS")

```


### Feature Selection
```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

# set.seed(100)
# options(warn=-1)
# 
# subsets <- c(1:5, 10, 15, 18)
# 
# ctrl <- rfeControl(functions = rfFuncs,
#                    method = "repeatedcv",
#                    repeats = 5,
#                    verbose = FALSE)
# 
# lmProfile <- rfe(x=trainData[, 1:18], y=trainData$Purchase,
#                  sizes = subsets,
#                  rfeControl = ctrl)
# 
# lmProfile %>%  kable() %>% kable_styling() %>% print()

```


## Escolhendo o algoritmo {.tabset}

### logistic regression

```{r, echo=FALSE, message=FALSE, warning=FALSE, results="asis"}

gbm <- train(Purchase~., data = trainData, 
                 method = "gbm", 
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)



gbm$results %>% kable() %>% kable_styling() %>% print()

plot(gbm, main="Model Accuracies with MARS")

varimp_mars <- varImp(gbm)
plot(varimp_mars, main="Variable Importance with MARS")
```


### Adaboost


### Random Forest


### xgBoost Dart


### SVM


## Comparando os modelos {.tabset}

### resamples

### Ensembling the predictions

##  combine the predictions of multiple models





