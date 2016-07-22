# Practical Machine Learning Project
J.E.Black  
July 21, 2016  

## Executive Summary


## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

Our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of six  participants. 
Each was asked to perform barbell lifts correctly and incorrectly in five different ways. 


## Exploratory Data Analysis

We acknowledge and thank the group who present their research on the website at: 
http://groupware.les.inf.puc-rio.br/har, 
from which we obtained the the data used in our analysis.
They have been very generous in allowing us to use their data.

- The training data used are are available from: 
"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

- The test data are available from: 
"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

We quote the authors' description of the data:

```
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg).
```
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4F6NogDKh


### Preparation

In order to set up the environment such that we can create
reproducible results, we need to load the necessary R libraries, and 
initialize a few parameters.


```r
library(caret)
library(knitr)
library(rattle)
library(randomForest)
library(rpart)
library(rpart.plot)
set.seed(527)
```

### Obtain Data

First check to see if we've already downloaded and saved a copy of the data;
if not, then download it from the public source in the internet.


```r
harPath <- "https://d396qusza40orc.cloudfront.net/predmachlearn/"
trnData <- "pml-training.csv"
tstData <- "pml-testing.csv"

if (!(file.exists(trnData))) { 
  download.file(paste(c(harPath,trnData),collapse=""), destfile=trnData)
}
if (!(file.exists(trnData))) {
  stop("Failed to download Training Data")
}
if (!(file.exists(tstData))) { 
  download.file(paste(c(harPath,tstData),collapse=""), destfile=tstData)
}
if (!(file.exists(tstData))) {
  stop("Failed to download Test Data")
}
```

### Pruning and Cleaning

Preliminary examination of the data indicated that there were 
numerous occurances of empty fields, and strings such as "NA" and "#DIV/0!" 
that were indicative of problems with the data.

We'll read the data from the "csv" files, and adjust the problem fields;
even in the test data, although the anomolies were observed only in the training data.
We just do this to make the data more uniform, so that it's easier 
to reduce the number of predictors in later steps.


```r
badStrings <- c("#DIV/0!", "NA", "")
trnFrame <- read.csv(trnData, na.strings=badStrings)
tstFrame <- read.csv(tstData, na.strings=badStrings)
```

Before we continue with additional cleaning of the data, 
we'll perform a few simple "sanity checks;" 
such as making sure that both data frames have the same number of columns, 
and that the columns have the same names in each data frame.
(except for "classe" and "problem_id" which are special cases applicable to the exercise)


```r
tstNames <- names(tstFrame)
trnNames <- names(trnFrame)
if (!(length(tstNames)==length(trnNames))) {
  stop("Mismatched column count")
}
tstNames[length(tstNames)] <- "X" # adjust last one
trnNames[length(trnNames)] <- "X"

if (!(identical(tstNames,trnNames))) {
  stop("Mismatched column names")
}
```

If we make it this far, then both the test data and the training data
are reasonably uniform, and we can continue processing.
For now, we'll set the Test Data aside, and work with the Training set.

We'll attempt to reduce the number of predictors by removing the columns
that we don't expect to influence the prediction, and then we'll split 
the training data into a training (60%) and a validation (40%) data set.

Currently we have 160 variables; 
we should be able to prune that down a bit by taking out
column that have near zero variance, and thus would not influence the prediction very much.


```r
nZ <- nearZeroVar(trnFrame)
trnFrame2 <- trnFrame[, -nZ]
```

Now we have only 124 columns.
Let's remove columns that are mostly "NA;" 
any column that is 90% or more "NA" is not likely to influence the prediction.


```r
mostlyNA <- sapply(trnFrame2, function(x) mean (is.na(x))) > 0.9
trnFrame2 <- trnFrame2[, mostlyNA==FALSE]
```
By removing the "mostly NA" columns, we've reduced the count to 59.

Although it is possible that the time of day that the exercise was performed
could have some influence on the outcome, we don't consider that it would have 
a significant effect on the prediction model.
Similarly, we couldn't justify that a window would affect the outcome.
Therefore, in order to further reduce the number of predictors, 
we'll remove columns 1 through 6, which appear to be a sample number, 
the subject's name, a couple of time stamps, and "num_window."


```r
trnFrame2 <- trnFrame2[, -(1:6)]
dim(trnFrame2)
```

```
## [1] 19622    53
```

Now we've reduced the number of variables to 53;
however we still have 19622 rows in the data set.














