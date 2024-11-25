library(ggplot2)

bike_data <- read.csv("bike.csv")
str(bike_data)

sum(is.na(bike_data))
library(keras3)

FLAGS <- flags(
  flag_numeric("nodes1", 64, "Number of nodes in the first hidden layer"),
  flag_numeric("nodes2", 32, "Number of nodes in the second hidden layer"),
  flag_numeric("batch_size", 32, "Batch size for training"),
  flag_numeric("learning_rate", 0.001, "Learning rate for the optimizer"),
  flag_string("activation", "relu", "Activation function for tuning")
)

# Prepare data
# Convert data frames to matrices (exclude target column 'count') and ensure numeric data type
x_train <- as.matrix(training_data[, !names(training_data) %in% c("count")])
y_train <- as.matrix(training_data$count)
x_val <- as.matrix(validation_data[, !names(validation_data) %in% c("count")])
y_val <- as.matrix(validation_data$count)

# Convert all data to numeric format
x_train <- matrix(as.numeric(x_train), nrow = nrow(x_train))
y_train <- matrix(as.numeric(y_train), nrow = nrow(y_train))
x_val <- matrix(as.numeric(x_val), nrow = nrow(x_val))
y_val <- matrix(as.numeric(y_val), nrow = nrow(y_val))

# Build the neural network model
model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = FLAGS$activation, input_shape = ncol(x_train)) %>%
  layer_dense(units = FLAGS$nodes2, activation = FLAGS$activation) %>%
  layer_dense(units = 1)  # Output layer with single neuron for regression task

# Compile the model with selected hyperparameters
model %>% compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learning_rate),
  loss = "mse",
  metrics = list("mean_squared_error")
)

# Fit the model
history <- model %>% fit(
  x = x_train,
  y = y_train,
  validation_data = list(x_val, y_val),
  epochs = 50,
  batch_size = FLAGS$batch_size
)

model
