card <- read.csv(file.choose(), header = T)
###############################################################
# Packages
###############################################################
library(rpart) #Decision tree
library(rpart.plot) #Decision tree plotting.
library(randomForest) #Random Forest
library(caret) #Data Processing
library(adabag) #Boosting 1
library(ada) #Boosting 2
library(corrplot) #Correlations
library(PRROC) # receiver operating characteristic curves
library(psych) #Calculate Cohen's Kappa Statistic
library(rlist)
###############################################################
str(card)
###############################################################
# Create a Factor
###############################################################
card$Class <- factor(card$Class, levels = c("1", "0"), labels = c("Fraud", "No Fraud"))

###############################################################
# Summary Stats
###############################################################
t1 <- table(card$Class)
# Fraud No Fraud 
# 492   284315 

t2 <- table(card$Class) / length(card$Class)
# Fraud    No Fraud 
# 0.001727486 0.998272514 


barchart <- barchart(table(card$Class)) #The data is skewed
 
barplot(table(card$Class))

boxplot(card$Amount ~ card$Class, horizontal = T)

boxplot <- boxplot(card$Time ~ card$Class, horizontal = T)

#All of these plots are skewed right (Most of the data is on left)

hist(card$Amount[card$Class == "No Fraud"])
hist(card$Amount[card$Class == "Fraud"])
plot(density(card$Amount[card$Class == "No Fraud"]), xlim = c(0, 1000), main = "")
points(density(card$Amount[card$Class == "Fraud"]), type = "l")


a.zero <- which(card$Amount <= 0) #Find indexes that contain zeros 
a.zero.card <- card[-a.zero, ] #remove these indexes
hist(log(a.zero.card$Amount[card$Class == "No Fraud"]), main = "No Fraud")
hist(log(a.zero.card$Amount[card$Class == "Fraud"]), main = "Fraud")

hist(card$Time[card$Class == "No Fraud"])
hist(card$Time[card$Class == "Fraud"]) #No recognizable difference in the two

#################################################################
#First, let's split the data
set.seed(702)
#################################################################
# From the cart package: Split the data into 80% Train - 20% test
#################################################################
trainIndex <- createDataPartition(card$Class, p = .8, 
                                                list = FALSE, 
                                                times = 1)

cardTrain <- card[trainIndex, ]
cardTest <- card[-trainIndex, ]

table(cardTrain$Class)
table(cardTest$Class)

table(card$Class)
#Verify it was done correctly
table(cardTrain$Class)[1] + table(cardTest$Class)[1] == table(card$Class)[1] 
# Fraud
# TRUE
table(cardTrain$Class)[2] + table(cardTest$Class)[2] == table(card$Class)[2]
# No Fraud 
# TRUE
############### Undersampling
undersample <- sample(which(cardTrain$Class == "No Fraud"), 1000)
unTrain <- rbind(cardTrain[cardTrain$Class == "Fraud",]
                , cardTrain[undersample,])
table(unTrain$Class)

#######################################################################
# The original unweighted tree
########################################################################
#Without xval
#Determine optimal cross validation
c.5 <- rpart.control(xval = 5, cp = 0)
c.10 <- rpart.control(xval = 10, cp = 0)
c.15 <- rpart.control(xval = 15, cp = 0)
c.20 <- rpart.control(xval = 20, cp = 0)
c.25 <- rpart.control(xval = 25, cp = 0)
c.30 <- rpart.control(xval = 30, cp = 0)
c.35 <- rpart.control(xval = 35, cp = 0)

#
no.c5 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.5)
no.c10 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.10) 
no.c15 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.15) 
no.c20 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.20) 
no.c25 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.25) 
no.c30 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.30) 
no.c35 <- rpart(cardTrain$Class ~., data = cardTrain, control = c.35) 

list.save(no.c5, "~/Desktop/DataMiningProj/no.c5.rds")
list.save(no.c10, "~/Desktop/DataMiningProj/no.c10.rds")
list.save(no.c15, "~/Desktop/DataMiningProj/no.c15.rds")
list.save(no.c20, "~/Desktop/DataMiningProj/no.c20.rds")
list.save(no.c25, "~/Desktop/DataMiningProj/no.c25.rds")
list.save(no.c30, "~/Desktop/DataMiningProj/no.c30.rds")
list.save(no.c35, "~/Desktop/DataMiningProj/no.c35.rds")
#Plot minimum cross validated error vs. V
err.x <- rep(NA, 7)
se.5 <- no.c5$cptable[which.min(no.c5$cptable[,4]),4] + no.c5$cptable[which.min(no.c5$cptable[,4]),5]
err.x[1] <- no.c5$cptable[which.max(no.c5$cptable[,4] < se.5), 4]

se.10 <- no.c10$cptable[which.min(no.c10$cptable[,4]),4] + no.c10$cptable[which.min(no.c10$cptable[,4]),5]
err.x[2] <- no.c10$cptable[which.max(no.c10$cptable[,4] < se.10), 4]

se.15 <- no.c15$cptable[which.min(no.c15$cptable[,4]),4] + no.c15$cptable[which.min(no.c15$cptable[,4]),5]
err.x[3] <- no.c15$cptable[which.max(no.c15$cptable[,4] < se.15), 4]

se.20 <- no.c20$cptable[which.min(no.c20$cptable[,4]),4] + no.c20$cptable[which.min(no.c20$cptable[,4]),5]
err.x[4] <- no.c20$cptable[which.max(no.c20$cptable[,4] < se.20), 4]

se.25 <- no.c25$cptable[which.min(no.c25$cptable[,4]),4] + no.c25$cptable[which.min(no.c25$cptable[,4]),5]
err.x[5] <- no.c25$cptable[which.max(no.c25$cptable[,4] < se.25), 4]

se.30 <- no.c30$cptable[which.min(no.c30$cptable[,4]),4] + no.c30$cptable[which.min(no.c30$cptable[,4]),5]
err.x[6] <- no.c30$cptable[which.max(no.c30$cptable[,4] < se.30), 4]

se.35 <- no.c35$cptable[which.min(no.c35$cptable[,4]),4] + no.c35$cptable[which.min(no.c35$cptable[,4]),5]
err.x[7] <- no.c35$cptable[which.max(no.c35$cptable[,4] < se.35), 4]

V <- seq(5, 35, by = 5)
cbind(V, err.x)

plot(V, err.x, xlab = "V-Fold Cross Validation", ylab = "Cross Validated Error")
lines(V, err.x, col = "red")

rm(no.c5)
rm(no.c10)
rm(no.c15)
rm(no.c20)
rm(no.c25)
rm(no.c35)

# It appears that 30 fold cross validation yields the smallest xval error

# > cbind(s, err.x)
# s     err.x
# [1,]  5 0.3527919
# [2,] 10 0.3527919
# [3,] 15 0.3071066
# [4,] 20 0.3121827
# [5,] 25 0.3147208
# [6,] 30 0.2918782
# [7,] 35 0.2969543



#Choose 25 Fold 

list.load("~/Desktop/DataMiningProj/no.c25.rds")
rpart.plot(no.c25)
no.c25p <- prune(no.c25, cp = 0.0013)
rpart.plot(no.c25p)
fold25p <- predict(no.c25p, newdata = cardTest[,-31], type = "prob")
Prediction.25p <- ifelse(fold25p[,1] > .5, "Fraud", "No Fraud")
Prediction.25.t <- ifelse(fold25[,1] > .05, "Fraud", "No Fraud")


table(Truth = cardTest$Class, Prediction.25p) # 0.5 Threshold

# Prediction.30
# Truth      Fraud No Fraud
# Fraud       76       22
# No Fraud     9    56854


#Changing threshold to minimize type II error
table(Truth = cardTest$Class, Prediction.30.t) # 0.05 Threshold
# Prediction.30.t
# Truth      Fraud No Fraud
# Fraud       83       15
# No Fraud    21    56842
 
cohen.kappa(table(Truth = cardTest$Class, Prediction.30))

plot(density(fold30[which(cardTest[,31] == "No Fraud"),2]), 
     main = "Density of probabilities")
points(density(fold30[which(cardTest[,31] == "Fraud"),2]), type = "l", col = "red")
plot(density(fold30[which(cardTest[,31] == "Fraud"),2],  bw="ucv"), col = "red", main = "Hist of 30-Fold tree prediction" ,xlab = "Prob")

#########################################################################

#With control to determine the best tree according to 1 SE rule
rpart.plot(no.c30)

printcp(no.c30) 
# Classification tree:
#   rpart(formula = cardTrain$Class ~ ., data = cardTrain, control = c.30)
# 
# Variables actually used in tree construction:
#   [1] V10 V12 V14 V16 V17 V2  V23 V25 V26 V27
# 
# Root node error: 394/227846 = 0.0017292
# 
# n= 227846 
# 
# CP nsplit rel error  xerror     xstd
# 1 0.46954315      0   1.00000 1.00000 0.050336
# 2 0.07868020      1   0.53046 0.53807 0.036938
# 3 0.05076142      2   0.45178 0.45685 0.034038
# 4 0.02199662      3   0.40102 0.41117 0.032293
# 5 0.01649746      6   0.33503 0.40102 0.031892
# 6 0.01522843      9   0.27919 0.38325 0.031178
# 7 0.00126904     10   0.26396 0.29188 0.027211
# 8 0.00084602     12   0.26142 0.28934 0.027092
# 9 0.00000000     15   0.25888 0.29442 0.027329

# 10 splits and 11 terminal nodes
0.28934 +0.027092
# [1] 0.316432
n.30 <- prune(no.c30, cp = 0.00127)
rpart.plot(n.30)
n.30.pred <- predict(n.30, newdata = cardTest[,-31])
n.30.pred <- ifelse(n.30.pred[,1] > .5, "Fraud", "No Fraud")
n.30.pred <- factor(Prediction, levels = c("Fraud", "No Fraud"))
cohen.kappa(table(Truth = cardTest$Class, n.30.pred))
# Call: cohen.kappa1(x = x, w = w, n.obs = n.obs, alpha = alpha, levels = levels)
# 
# Cohen Kappa and Weighted Kappa correlation coefficients and confidence boundaries 
# lower estimate upper
# unweighted kappa  0.76     0.82  0.88
# weighted kappa    0.76     0.82  0.88
# 
# Number of subjects = 56961 

#########################################################################
#According to the table the best tree is the one with 11 terminal nodes
#Best tree based on 1 SE RULE
#########################################################################

##################################
# Add a misclassification weight
##################################
# Want to minimize classifying not fraud when it is
# Thus, we want to minimize type two error

lmat <- matrix(c(0,1, 500,0), nrow = 2, byrow = T)
# [,1] [,2]
# [1,]  0    50
# [2,]  1    0
tree1 <- rpart(Class ~. , data = cardTrain, control = c.25, parms = list(loss = matrix(c(0,100,1,0), nrow = 2)))
list.save(tree1, "~/Desktop/DataMiningProj/tree1.rds")
rpart.plot(tree1)
tree1$cptable

tree.pred <- predict(tree1, newdata = cardTest[,-31])
Prediction <- ifelse(tree.pred[,1] > .5, "Fraud", "No Fraud")
Prediction <- factor(Prediction, levels = c("Fraud", "No Fraud"))

table(Truth = cardTest$Class, Prediction)


utree <- rpart(Class ~. , data = unTrain, control = c.10, parms = list(loss = matrix(c(0,100,1,0), nrow = 2)))
utreep <- prune(utree, cp = .01)
rpart.plot(utreep)

utree.predict <- predict(utreep, newdata = cardTest)
utree.predict<- ifelse(utree.predict [,1] > .5, "Fraud", "No Fraud")

# It confusion is a little better, but the agreement is the same!
cohen.kappa(table(Truth = cardTest$Class, Prediction))

which.min(tree1$cptable[,4])
# 1
#Pruning lowers this to only the root node!
tree1.13 <- prune(tree1, cp = .77)
rpart.plot(tree1.13)

tree.pred13 <- predict(tree1.13, newdata = cardTest[,-31])
Prediction13 <- ifelse(tree.pred13[,1] > .5, "Fraud", "No Fraud")
Prediction13 <- factor(Prediction13, levels = c("Fraud", "No Fraud"))

table(Truth = cardTest$Class, Prediction13)


cohen.kappa(table(Truth = cardTest$Class, Prediction13))
 


###############################################################
# We can possibly increse the weights
# Increase weights
###############################################################
lmat2 <- matrix(c(0,1, 100,0), nrow = 2, byrow = T) #Consider the following loss matrix
# [,1] [,2]
# [1,]    0    100
# [2,]    1    0
tree2 <- rpart(Class ~. , data = cardTrain, control = c.25, parms = list(loss = lmat2))
list.save(tree2, "~/Desktop/DataMiningProj/tree2.rds")

rpart.plot(tree2)
rm(tree2)
tree2.pred <- predict(tree2, newdata = cardTest[,-31])
Prediction2 <- ifelse(tree2.pred[,1] > .5, "Fraud", "No Fraud")
Prediction2 <- factor(Prediction2, levels = c("Fraud", "No Fraud"))

table(Truth = cardTest$Class, Prediction2)

# Prediction2
# Truth      Fraud No Fraud
# Fraud       80       18
# No Fraud    14    56849

which.min(tree2$cptable[,4])
#1

#SAME RESULTS AS BEFORE
###########################################################################
# ROC Curve for goodness of fit
############################################################################
fg.t <- tree.pred13[cardTest$Class == "Fraud"]
bg.t <- tree.pred13[cardTest$Class == "No Fraud"]

roc.t <- roc.curve(scores.class0 = fg.t, scores.class1 = bg.t, curve = T)

list.save(roc.t, "~/Desktop/DataMiningProj/roc.t.rds")
plot(roc.t)               # Does not do well at all. Just randomly guesses


#Therefore, we need another method besides trees
################################################################################
# Random forests
#################################################################################
rF1 <- randomForest(Class ~., data = cardTrain, ntree = 1000)
rF2 <- randomForest(Class ~., data = cardTrain, ntree = 1000)


rF1 <- list.load("~/Desktop/DataMiningProj/rF1.rds")
rF.pred <- predict(rF1, newdata = cardTest[,-31])
rF.pred.prob <- predict(rF1, newdata = cardTest[,-31], type = "prob")
rF.pred <- factor(rF.pred, c("Fraud", "No Fraud"))
table(True = cardTest$Class, rF.pred)

#

#


which((rF.pred == "No Fraud" )& (cardTest[,31] == "Fraud"))
# [1]  8706 43094

par(mfrow= c(1,2))
plot(density(rF.pred.prob[which(cardTest[,31] == "Fraud"),1]), main = "Density of probabilities RF")
points(density(rF.pred.prob[which(cardTest[,31] == "No Fraud"),1]), type = "l", col = "red")

plot(density(rF.pred.prob[which(cardTest[,31] == "No Fraud"),1]),
     main = "Density of probabilities RF", col = "red")
points(density(rF.pred.prob[which(cardTest[,31] == "Fraud"),1]), type = "h", col = "black")
par(mfrow= c(1,1))


rF.newpred <- ifelse(rF.pred.prob[,2] > 0.1, "Fraud", "No Fraud" )
table(True = cardTest$Class, rF.newpred)
# #           rF.newpred
# True       Fraud No Fraud
# Fraud       97        1
# No Fraud     1    56862

rF.newpred <- ifelse(rF.pred.prob[,2] > 0.45, "Fraud", "No Fraud" )
table(True = cardTest$Class, rF.newpred)
# rF.newpred
# True       Fraud No Fraud
# Fraud       97        1
# No Fraud     1    56862

#Good Balance

#Nearly perfect separation between density plots

cohen.kappa(table(True = cardTest$Class, rF.pred)) 
# 
# Call: cohen.kappa1(x = x, w = w, n.obs = n.obs, alpha = alpha, levels = levels)
# 
# Cohen Kappa and Weighted Kappa correlation coefficients and confidence boundaries 
# lower estimate upper
# unweighted kappa  0.98     0.99     1
# weighted kappa    0.98     0.99     1
# 
# Number of subjects = 56961 


#########
# ROC rF
#########

fg <- rF.pred.prob[cardTest$Class == "Fraud", 2] #Consider fraud in the foreground
bg <- rF.pred.prob[cardTest$Class == "No Fraud", 2] #No Fraud as background

roc.rF <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
list.save(roc.rF, "~/Desktop/DataMiningProj/rocrF.rds")

plot(roc.rF) 

#Var Importantace
varImpPlot(rF1)

list.save(rF1, "~/Desktop/DataMiningProj/rF1.rds")
rm(rF1)

#################################################################################################
# Boosting Algorithm
#################################################################################################
#Stumps via Discrete
####################################################################
stump <- rpart.control(maxdepth=1, cp=-1, minsplit=0, xval=0) #No cross validation, only split once

boost1 <- ada(Class ~., data = cardTrain, iter = 1000, type="discrete", 
              control = stump) #Build he tree
boost1<- list.load("~/Desktop/DataMiningProj/boost1.rds") #Save the data to desktop

boost.compare <- predict(boost1, newdata = cardTest[,-31]) 

boost.compare <- factor(boost.compare, c( "Fraud", "No Fraud"))

table(True = cardTest$Class, boost.compare)

# True       Fraud No Fraud
# Fraud       83       15
# No Fraud     3    56860


boost.compare.prob <- predict(boost1, newdata = cardTest[,-31], type = "prob") 
boost.compare.prob <- ifesle(boost.compare.prob[,1] > .01, "Fraud", "No Fraud")
boost.compare.prob <- factor(boost.compare.prob, c( "Fraud", "No Fraud"))

table(True = cardTest$Class, boost.compare.prob)

plot(density(boost.compare.prob[which(cardTest[,31] == "Fraud"),1]), main = "Density of probabilities Boosingt")
points(density(boost.compare.prob[which(cardTest[,31] == "No Fraud"),1]), type = "l", col = "red")


un.boost1 <- ada(Class ~., data = unTrain, iter = 1000, type="discrete", 
                 control = stump) 
un1.pred <- predict(un.boost1, newdata = cardTest)
table(True = cardTest$Class, un1.pred)


##############
# ROC Boosting
##############

fg <- boost.compare.prob$prob[cardTest$Class == "Fraud", 1]
bg <- boost.compare.prob$prob[cardTest$Class == "No Fraud", 1]

roc.boost <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc.boost) 
list.save(roc.boost, "~/Desktop/DataMiningProj/rocboost.rds")


importanceplot(boost1)
list.save(boost1, "~/Desktop/DataMiningProj/boost1.rds")
rm(boost1)
####################################################################
# Discrete ada Boost
####################################################################
boost2 <- ada(Class ~., data = cardTrain, iter = 100, type="discrete")
list.save(boost2, "~/Desktop/DataMiningProj/boost2.rds")
boost2 <- list.load("~/Desktop/DataMiningProj/boost2.rds")
boost2.pred <- predict(boost2, newdata = cardTest[,-31])
boost2.pred.prob <- predict(boost2, newdata = cardTest[,-31], type = "prob")

boost2.pred <- factor(boost2.pred, c("Fraud", "No Fraud"))

boost2.pred.prob <- ifelse(boost2.pred.prob[,1] > 0.01, "Fraud", "No Fraud")
boost2.pred.prob <- factor(boost2.pred.prob, c("Fraud", "No Fraud"))

table(True = cardTest$Class, boost2.pred)

# True       Fraud No Fraud
# Fraud       83       15
# No Fraud     1    56862

table(True = cardTest$Class, boost2.pred.prob)

newboost <- addtest(boost2, cardTest[,-31], cardTest[,31])
plot(newboost, test = T)
rm(boost2)

cohen.kappa(table(True = cardTest$Class, boost2.pred))

# Call: cohen.kappa1(x = x, w = w, n.obs = n.obs, alpha = alpha, levels = levels)
# 
# Cohen Kappa and Weighted Kappa correlation coefficients and confidence boundaries 
# lower estimate upper
# unweighted kappa  0.87     0.91  0.95
# weighted kappa    0.87     0.91  0.95


fg <- boost2.pred.prob[cardTest$Class == "Fraud", 1] #Positive (Foreground)
bg <- boost2.pred.prob[cardTest$Class == "No Fraud", 1] #Negative(Background)

roc.boost2 <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
list.save(roc.boost2, "~/Desktop/DataMiningProj/rocboost2.rds")

plot(roc.boost2) 

####################################################################
# Real ada Boost
####################################################################
boost3 <- ada(Class ~., data = cardTrain, iter = 200, type="real")
boost3.t <- ada(Class ~., data = cardTrain, iter = 10, type="real")

list.save(boost3, "~/Desktop/DataMiningProj/boost3.rds")

boost3.pred <- predict(boost3, newdata = cardTest[,-31])

boost3.pred.prob <- predict(boost3, newdata = cardTest[,-31], type = "prob")

boost3.pred <- factor(boost3.pred, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost3.pred)
# boost3.pred
# True       Fraud No Fraud
# Fraud       80       18
# No Fraud     6    56857

newtest <- addtest(boost3, cardTest[,-31], cardTest[,31])
plot(newtest, test = T)

cohen.kappa(table(True = cardTest$Class, boost3.pred))

# Call: cohen.kappa1(x = x, w = w, n.obs = n.obs, alpha = alpha, levels = levels)
# 
# Cohen Kappa and Weighted Kappa correlation coefficients and confidence boundaries 
# lower estimate upper
# unweighted kappa  0.82     0.87  0.93
# weighted kappa    0.82     0.87  0.93
# 
# Number of subjects = 56961 

fg.3 <- boost3.pred.prob[cardTest$Class == "Fraud", 1] #Positive (Foreground)
bg.3 <- boost3.pred.prob[cardTest$Class == "No Fraud", 1] #Negative(Background)
roc.boost3 <- roc.curve(scores.class0 = fg.3, scores.class1 = bg.3, curve = T)
list.save(roc.boost3, "~/Desktop/DataMiningProj/rocboost3.rds")
boost3 <- list.load("~/Desktop/DataMiningProj/boost3.rds")


plot(roc.boost3) 
rm(boost3)
####################################################################
# Gentle ada Boost
####################################################################
boost4 <- ada(Class ~., data = cardTrain, iter = 200, type="gentle")
list.save(boost4, "~/Desktop/DataMiningProj/boost4.rds")

boost4.pred <- predict(boost4, newdata = cardTest[,-31])

boost4.pred <- factor(boost4.pred, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost4.pred)
# boost4.pred
# True       Fraud No Fraud
# Fraud       83       15
# No Fraud     6    56857

cohen.kappa(table(True = cardTest$Class, boost4.pred))

boost4 <- list.load("~/Desktop/DataMiningProj/boost4.rds")
# Call: cohen.kappa1(x = x, w = w, n.obs = n.obs, alpha = alpha, levels = levels)
# 
# Cohen Kappa and Weighted Kappa correlation coefficients and confidence boundaries 
# lower estimate upper
# unweighted kappa  0.84     0.89  0.94
# weighted kappa    0.84     0.89  0.94
# 
# Number of subjects = 56961 

##
# ROC CURVE
#
newboost <- addtest(boost4, cardTest[,-31], cardTest[,31])
plot(newboost, test = T)
print(boost4)
# boost with weight
#
lmat <- matrix(c(0, 50, 1, 0), nrow = 2, byrow = T) #False Positive? or False Negative? #Says fP want FN
#boost5 <- ada(Class ~., data = cardTrain, iter = 100, type="real", parms = list(loss = lmat.t))
list.save(boost5, "~/Desktop/DataMiningProj/boost5.rds")
rm(boost5)
lmat.t <- matrix(c(0, 1, 50, 0), nrow = 2, byrow = T) #False Positive? or False Negative? #Says fP want FN
#boost5.t <- ada(Class ~., data = cardTrain, iter = 100, type="discrete" ,parms = list(loss = lmat.t))
list.save(boost5.t, "~/Desktop/DataMiningProj/boost5t.rds")
rm(boost5.t)
boost5e <- ada(Class ~., data = cardTrain, iter = 100, type="real", 
               parms = list(loss = lmat.t), loss = "e",
               control=rpart.control(cp=-1,maxdepth=2))
list.save(boost5e, "~/Desktop/DataMiningProj/boost5e.rds")
rm(boost5e)

boost5el <- ada(Class ~., data = cardTrain, iter = 70, type="gentle", 
               parms = list(prior = c(0.9999999999999, 1 - 0.9999999999999)),
               loss = "e", control=rpart.control(cp=-1, maxdepth=8)) 
boost5elp <- ada(Class ~., data = cardTrain, iter = 70, type="gentle", 
                parms = list(prior = c(0.9999999999999, 1 - 0.9999999999999), loss = matrix(c(0, 1, 1000, 0))),
                loss = "e", control=rpart.control(cp=-1, maxdepth=8)) 
boostun <- ada(Class ~., data = unTrain, iter = 500, type="gentle", 
    parms = list(prior = c(0.99, 1 - 0.99)),
    loss = "e", control=rpart.control(cp=-1, maxdepth=3)) 

newboost <- addtest(boost5elp, cardTest[,-31], cardTest[,31])

boost5ep <- ada(Class ~., data = cardTrain, iter = 70, type="gentle", 
                parms = list(prior = c(0.9999999999999, 1 - 0.9999999999999)),
                loss = "e", control=rpart.control(maxdepth=8))
boost5ep.prdict <- predict(boost5ep, newdata = cardTest[,-31])

table(cardTest[,31],boost5ep.prdict)

boost5el.prdict <- predict(boost5el, newdata = cardTest[,-31])
boost5el.prdict.prob <- predict(boost5el, newdata = cardTest[,-31], type = "prob")

table(cardTest[,31],boost5el.prdict)

gen2 <- addtest(boost5el,cardTest[,-31],cardTest[,31])
summary(gen2) 
plot(gen2,TRUE,TRUE)
varplot(gen2)

gen1 <- ada(Class ~., data = cardTrain, loss = "e",
            type="gentle",control=rpart.control(cp=-1,maxdepth=8, xval = 0), iter=70, 
            parms = list(loss =matrix(c(0, 1, 500, 0))))
gen2 <- ada(Class ~., data = cardTrain, loss = "e",
            type="gentle",control=rpart.control(cp=-1,maxdepth=8, xval = 0), iter=70, 
            parms = list(loss =matrix(c(0, 1, 1000, 0))))
            
gen23 <- addtest(gen1,cardTest[,-31],cardTest[,31])

plot(gen23)

#boost5.pred <- predict(boost5, newdata = cardTest[,-31])

boost5.pred <- factor(boost5.pred, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.pred)

boost5.pred.t <- predict(boost5.t, newdata = cardTest[,-31])

boost5.pred.t <- factor(boost5.pred.t, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.pred.t)

boost5.prede <- predict(boost5e, newdata = cardTest[,-31])

boost5.prede <- factor(boost5.prede, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.prede)


boost5.predel <- predict(boost5el, newdata = cardTest[,-31])

boost5.predel <- factor(boost5.predel, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.predel)

boost5.gen <- predict(gen1, newdata = cardTest[,-31])

boost5.gen <- factor(boost5.gen, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.gen)

boost5.gen2 <- predict(gen2, newdata = cardTest[,-31], type = "prob")
boost5.gen2.pred <- ifelse(boost5.gen2[,1] > .05, "Fraud", "No Fraud")
boost5.gen2.pred <- factor(boost5.gen2.pred, c("Fraud", "No Fraud"))
table(True = cardTest$Class, boost5.gen2.pred)

print(boost4)
# How do stumps measure up?
#

#########################################################################
# Variable importance
#########################################################################
par(mfrow = c(1,4))
varImpPlot(rF1)
varplot(boost2)
varplot(boost3)
varplot(boost4)
par(mfrow = c(1,1))


#First 4 Variables rF 

par(mfrow = c(2,2))
plot(density(card$V17), type = "n", main = "Density V17")
points(density(card$V17[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V17[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V12), type = "n", main = "Density V12")
points(density(card$V12[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V12[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V14), type = "n", main = "Density V14")
points(density(card$V14[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V14[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V10), type = "n", main = "Density V10")
points(density(card$V10[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V10[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))
par(mfrow = c(1,1))

#Boost2

par(mfrow = c(2,2))
plot(density(card$V7), type = "n", main = "Density V7")
points(density(card$V7[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V7[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V16), type = "n", main = "Density V16")
points(density(card$V16[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V16[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))


plot(density(card$V2), type = "n", main = "Density V2")
points(density(card$V2[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V2[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))
par(mfrow = c(1,1))
#Boost3

par(mfrow = c(2,2))
plot(density(card$V5), type = "n", main = "Density V5")
points(density(card$V5[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V5[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V1), type = "n", main = "Density V1")
points(density(card$V1[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V1[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))



plot(density(card$V10), type = "n", main = "Density V10")
points(density(card$V10[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V10[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

plot(density(card$V24), type = "n", main = "Density V24")
points(density(card$V24[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$V24[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))
par(mfrow = c(1,1))


par(mfrow = c(1,2))
# Time
plot(density(card$Time), type = "n", main = "Density Time")
points(density(card$Time[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$Time[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topleft", c("Fraud", "No Fraud"), fill = c("red", "blue"))

# Amount: Max is 25691.16 (No Fraud)
plot(density(card$Amount), type = "n", main = "Density Amount", xlim = c(-1, 1000))
points(density(card$Amount[card$Class == "Fraud"]), col = "red", type = "l")
points(density(card$Amount[card$Class == "No Fraud"]), col = "blue", type = "l")
legend("topright", c("Fraud", "No Fraud"), fill = c("red", "blue"))
par(mfrow = c(1,1))

# correlations
correlations <- cor(cardTrain[,-31])
corrplot(correlations, method="square")
