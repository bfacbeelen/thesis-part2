# Load required libraries
library(tidyverse)
library(caret)
library(e1071)         # For SVM
library(randomForest)
library(gbm)           # Gradient Boosting
library(xgboost)       # XGBoost
library(MLmetrics)
library(PerformanceAnalytics)
library(nortest)
library(car)
library(factoextra)
library(ggplot2)
library(tidyr)

# Set seed using your student ID
set.seed(311441)

# Load the dataset
data <- read.csv("/Users/bentebeelen/documents/shopping_behavior_updated.csv")
summary(data)

# Convert target and relevant predictors to factors
data$Discount.Applied <- as.factor(data$Discount.Applied)
data$Promo.Code.Used <- as.factor(data$Promo.Code.Used)
data$Shipping.Type <- as.factor(data$Shipping.Type)
data$Frequency.of.Purchases <- as.factor(data$Frequency.of.Purchases)
data$Payment.Method <- as.factor(data$Payment.Method)
data$Gender_numeric <- ifelse(data$Gender == "Male", 1, 
                            ifelse(data$Gender == "Female", 0, NA))

# Define the outcome and predictors
data_model <- data %>%
  select(Purchase.Amount..USD., Discount.Applied, Promo.Code.Used, Shipping.Type, Previous.Purchases, Gender_numeric, Age, Frequency.of.Purchases, Payment.Method)

# Split the dataset
trainIndex <- createDataPartition(data_model$Purchase.Amount..USD., p = 0.6, list = FALSE)
train <- data_model[trainIndex, ]
temp <- data_model[-trainIndex, ]
validIndex <- createDataPartition(temp$Purchase.Amount..USD., p = 0.5, list = FALSE)
valid <- temp[validIndex, ]
test <- temp[-validIndex, ]

# ---- LINEAR REGRESSION ----
lm_model <- lm(Purchase.Amount..USD. ~ ., data = train)
lm_preds <- predict(lm_model, newdata = test)
lm_results <- evaluate_model(test$Purchase.Amount..USD., lm_preds)
plot(lm_preds, test$Purchase.Amount..USD. - lm_preds, 
     main = "Residual Plot: Linear Regression", 
     xlab = "Predicted", ylab = "Residuals", col = "darkred", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- SVM ----
svm_model <- svm(Purchase.Amount..USD. ~ ., data = train)
svm_preds <- predict(svm_model, newdata = test)
svm_results <- evaluate_model(test$Purchase.Amount..USD., svm_preds)
plot(svm_preds, test$Purchase.Amount..USD. - svm_preds, 
     main = "Residual Plot: SVM", 
     xlab = "Predicted", ylab = "Residuals", col = "darkgreen", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- RANDOM FOREST ----
rf_model <- randomForest(Purchase.Amount..USD. ~ ., data = train)
rf_preds <- predict(rf_model, newdata = test)
rf_results <- evaluate_model(test$Purchase.Amount..USD., rf_preds)
plot(rf_preds, test$Purchase.Amount..USD. - rf_preds, 
     main = "Residual Plot: Random Forest", 
     xlab = "Predicted", ylab = "Residuals", col = "darkblue", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- GRADIENT BOOSTING ----
gbm_model <- gbm(Purchase.Amount..USD. ~ ., 
                 data = train, 
                 distribution = "gaussian", 
                 n.trees = 100)
gbm_preds <- predict(gbm_model, newdata = test, n.trees = 100)
gbm_results <- evaluate_model(test$Purchase.Amount..USD., gbm_preds)
plot(gbm_preds, test$Purchase.Amount..USD. - gbm_preds, 
     main = "Residual Plot: GBM", 
     xlab = "Predicted", ylab = "Residuals", col = "purple", pch = 20)
abline(h = 0, col = "blue", lty = 2)

# ---- XGBOOST ----
train_matrix <- model.matrix(Purchase.Amount..USD. ~ . -1, data = train)
test_matrix <- model.matrix(Purchase.Amount..USD. ~ . -1, data = test)
xgb_train <- xgb.DMatrix(data = train_matrix, label = train$Purchase.Amount..USD.)
xgb_test <- xgb.DMatrix(data = test_matrix, label = test$Purchase.Amount..USD.)

xgb_model <- xgboost(data = xgb_train, objective = "reg:squarederror", nrounds = 100, verbose = 0)
xgb_preds <- predict(xgb_model, newdata = xgb_test)
xgb_results <- evaluate_model(test$Purchase.Amount..USD., xgb_preds)

plot(xgb_preds, test$Purchase.Amount..USD. - xgb_preds, 
     main = "Residual Plot: XGBoost", 
     xlab = "Predicted", ylab = "Residuals", col = "orange", pch = 20)
abline(h = 0, col = "blue", lty = 2)
par(mfrow = c(1, 1))

# --- Output results ---
cat("=== Linear Regression ===\n")
print(lm_results)

cat("\n=== SVM ===\n")
print(svm_results)

cat("\n=== Random Forest ===\n")
print(rf_results)

cat("\n=== GBM ===\n")
print(gbm_results)

cat("\n=== XGBoost ===\n")
print(xgb_results)

# --- Correlation Plot ---
# Create working copy of correlation data
corr_data <- data_model[, c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases")]

# Ensure all are numeric and remove rows with NA
corr_data <- as.data.frame(sapply(corr_data, as.numeric))
corr_data <- na.omit(corr_data)
names(corr_data)[names(corr_data) == "Purchase.Amount..USD."] <- "Pn"

# Plot updated correlation matrix
chart.Correlation(corr_data, histogram = TRUE, pch = 19)

# --- Univariate Normality Tests ---
cat("\n=== Univariate Normality Tests ===\n")
num_vars <- sapply(df, is.numeric)
for (var in names(df)[num_vars]) {
  cat("\nVariable:", var, "\n")
  print(shapiro.test(df[[var]]))
  print(lillie.test(df[[var]]))
  print(ad.test(df[[var]]))
}

# --- Influence Diagnostics ---
cat("\n=== Influence Diagnostics ===\n")
influencePlot(lm_model, id.method = "identify", main = "Cook's D Bar Plot", sub = "Threshold: 4/n")
par(mfrow = c(2, 3))
for (var in names(coef(lm_model))) {
  plot(dfbetas(lm_model)[, var], type = "h", main = paste("Influence Diagnostics for", var),
       xlab = "Observation", ylab = "DFBETAS", col = "blue")
  abline(h = c(-1, 1) * 2 / sqrt(nrow(train)), col = "red", lty = 2)
}
plot(dffits(lm_model), type = "h", col = "blue", main = "Influence Diagnostics for Purchase Amount",
     xlab = "Observation", ylab = "DFFITS")
abline(h = c(-1, 1) * 2 * sqrt(length(coef(lm_model)) / nrow(train)), col = "red", lty = 2)
par(mfrow = c(1, 1))

# --- Clustering and Cluster Boxplots ---
cat("\n=== Clustering and Cluster Boxplots ===\n")
cluster_vars <- data_model[, c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases")]

# Ensure all columns are numeric and remove any rows with NA
cluster_vars <- as.data.frame(sapply(cluster_vars, as.numeric))
cluster_vars <- na.omit(cluster_vars)

# Now scale
cluster_vars_scaled <- scale(cluster_vars)

# K-means clustering
kmeans_result <- kmeans(cluster_vars_scaled, centers = 3, nstart = 25)
data_model$cluster <- as.factor(kmeans_result$cluster)

# PCA for visualization
pca_res <- prcomp(cluster_vars_scaled)
fviz_cluster(kmeans_result, data = cluster_vars_scaled, geom = "point", ellipse.type = "convex", 
             palette = c("#E41A1C", "#4DAF4A", "#377EB8"), ggtheme = theme_minimal())

# Cluster boxplots
cluster_long <- pivot_longer(
  data_model,
  cols = c("Purchase.Amount..USD.", "Previous.Purchases", "Gender_numeric", "Age", "Frequency.of.Purchases"),
  names_to = "var",
  values_to = "value",
  values_transform = list(value = as.numeric)
)
ggplot(cluster_long, aes(x = var, y = value, fill = cluster)) +
  geom_boxplot(outlier.shape = 1, position = position_dodge(width = 0.75)) +
  scale_fill_manual(values = c("#E41A1C", "#377EB8", "#4DAF4A")) +
  theme_minimal() +
  labs(title = "Boxplot by Cluster", x = "Variable", y = "Value")



# ---- EVALUATION ----
evaluate <- function(pred, true) {
  pred <- as.numeric(pred)
  true <- as.numeric(true)
  
  RMSE <- sqrt(mean((pred - true)^2, na.rm = TRUE))
  MAE <- mean(abs(pred - true), na.rm = TRUE)
  
  # Create default MAPE in case all true values are zero
  MAPE <- NA
  
  if (any(true != 0)) {
    nonzero <- true != 0
    mape_vals <- abs((pred[nonzero] - true[nonzero]) / true[nonzero])
    # Check for any NaN or Inf
    mape_vals <- mape_vals[is.finite(mape_vals)]
    if (length(mape_vals) > 0) {
      MAPE <- mean(mape_vals) * 100
    }
  }
  
  return(data.frame(RMSE = RMSE, MAE = MAE, MAPE = MAPE))
}


results <- list(
  Linear_Regression = evaluate(lm_pred, test$Purchase.Amount..USD.),
  SVM = evaluate(svm_pred, test$Purchase.Amount..USD.),
  Random_Forest = evaluate(rf_pred, test$Purchase.Amount..USD.),
  Gradient_Boosting = evaluate(gbm_pred, test$Purchase.Amount..USD.),
  XGBoost = evaluate(xgb_pred, test$Purchase.Amount..USD.)
)

print(results)
