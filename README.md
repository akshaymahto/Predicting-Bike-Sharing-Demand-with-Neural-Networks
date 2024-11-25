# I have completed this project during my First Semester where my  subjct was "Machine Learning with R" at University of illinois, Springfield.
# Predicting Bike Sharing Demand (6 points)
For this problem, you will be working with the bike sharing demand dataset from Kaggle. The dataset
is comprised of hourly bike rental data spanning two years (2011-2012). The goal is to predict the
total count of bike rentals based on features such as date, temperature, whether it is holiday, working
day, etc. This data is time-series because the observations are in a sequence and there is a temporal
order between them. However for the sake of this assignment, let’s assume that the observations are
independent and identically distributed (i.i.d). Click on the link above and read the data description
on Kaggle. And then download “bike.csv” file from canvas.
# 1. (0.4 points) Explore the overall structure of the dataset using the “str()” function. Then,
get a summary statistics of each variable using the “summary()” function. Answer the following
questions:
– How many observations (examples) do you have in the data?
– What is the type of each variable? Categorical or Numeric?
– Is there any missing value in the data?
– Draw the histogram of the variable “count”.
# 2. (0.1 points) Remove the “registered” and “casual” variables. These are the count of registered
and casual users and together they can perfectly predict “count” so we are removing them from
the model and predict count from the other features.
# 3. (0.1 points) The “count” variable is severely right-skewed. A skewed target variable can make
a machine learning model biased. For instance, in this case, lower counts are more frequent in
the data compared to higher counts . Therefore, a machine learning model trained on this data
is less likely to successfully predict higher counts. There are different ways we can transform a
right-skewed variable to a more bell-shape distribution. Common transformations for a right-
skewed data includes log, square-root, and cube-root transformations. We are going to use
square root transformation to make the distribution of count more bell-shaped. Set
the count variable in your dataframe equal to square root of count and plot its
histogram again.
# 4. (0.4 points) The variable of “datetime” is not useful in its current form. Convert this variable
to several variables: “year”, “month”, “day of month”, “day of week”, and “hour”. You can use
as.POSIXlt function to extract those features from “datetime”. Please see the following exam-
ple. After this, remove the original “datetime” variable after conversion by setting “datetime”
be NULL.
1
# Sample datetime
datetime <- as.POSIXlt("2024-11-01 12:04:17")
# Extract components as numeric values
year <- as.numeric(format(datetime, "%Y"))
month <- as.numeric(format(datetime, "%m"))
day_of_month <- as.numeric(format(datetime, "%d"))
day_of_week <- as.numeric(format(datetime, "%u"))
hour <- as.numeric(format(datetime, "%H"))
# 5. (0.8 points) Variables “month”, “day of week”, “hour”, and “season” are categorical but
they are also circular. This means these variables are periodic in nature. If we represent
a circular variable like “month” with numeric indices 0-11 we are implying that the distance
between month 10 (November) and month 11 (December) is much closer than the distance
between month 11(December) and month 0 (January) which is not correct. Thus, a better way
to represent these variables is to map each value into a point in a circle where the lowest value
appears next to the largest value in the circle. For instance, we can transform the “month i” by
creating “month i x” and “month i y” coordinates of the point in such the circle using sine and
cosine transformations as follows:
month i x = cos( 2π ∗ month i
max(month) )
month i y = sin( 2π ∗ month i
max(month) )
For instance, month 10 is converted to month 10 x = cos(2π ∗10/11) and month 10 y = sin(2π ∗
10/11).
Convert variables “month”, “day of week”, “hour”, and “season” to their x and y coordinates
using sine and cosine transformations as explained above. Make sure to remove the original
“month”, “day of week”, “hour”, and “season” variables after transformation by setting them as
NULL.
Note: The “day of month” variable is also technically circular but this dataset only contains
days 1-19. Therefore, we do not apply such the transformation on this variable.
# 6. (0.4 points) Encode the categorical variables before training the network. One-hot encode
all the categorical variables in your dataset, see your previous assignment and code demo to
see how you can do one-hot encoding for categorical variables. Note: binary variables such as
“holiday” and “workingday” have already been converted to binary 0-1 and don’t need to be
one-hot encoded. You can check their values to ensure they are binary. If not, you can use ifelse
to convert them to be binary.
# 7. (0 point) Use set.seed(2024) to set the random seed so that I can reproduce your results.
# 8. (0.4 points) Use Caret’s createDataPartition method as follows to split the dataset into
“bikes train” and “bikes test” (use 90% for training and 10% for testing).
inTrain = createDataPartition(bikes$count, p=0.9, list=FALSE)
bikes_train = bikes[inTrain,]
bikes_test = bikes[-inTrain,]
where “bikes” is the name of your pre-processed data frame. The first line creates a random 90%-
10% split of data such that the distribution of the target variable “bikes$count” is preserved in
each split. The “list = FALSE” option avoids returning the data as a list. Instead, “inTrain” is
a vector of indices used to get the training and test data.
2
# 9. (0.4 points) Set.seed(2024) and further divide the “bikes train” data into 90% training and 10%
validation using Caret’s “CreateDataPartition” function. This is for the later hyper-parameter
tuning.
# 10. (0.4 points) Scale the numeric variables in the training data (except for the target variable,
“count”). Use the column means and column standard deviations from the training data to scale
both the validation and test data (see code demo of week 10). Note: You should NOT scale the
categorical variables (one-hot-encoded or binary) in the data.
# 11. (1.6 points) In this part, we want to build a two-hidden layer neural network to predict the
total counts of bike rentals, fine-tune the hyper-parameters of this model, and find the best set
of hyper-parameters that can give us the best validation performance.
Specifically, we create an R script and name it “bike model.R”. In this script, we define a set of
flags for hyper-parameters we want to tune, including
– the number of nodes in hidden layer 1
– the number of nodes in hidden layer 2
– the activation functions
– the batch size
– the learning rate
Then, we build, compile, and fit the model as usual. You might want to see a concrete example
in the code demo of week 10 (“fashion mnist.R”).
After setting this script, go back to your original R notebook, run “tuning run” from “tfruns”
package to fine-tune the above hyper-parameters. As for the candidate values of your hyper-
parameters, you can determine based on your preference, and you can check one concrete example
in the code demo of week 10.
Finally, print the returned value from “tuning run” to see the metrics for each run. Which run
(which hyper-parameter combination) gives the best mean squared error on the validation set?
And print the learning curve for your best model. You can just take a screenshot of the learning
curve of your best model and submit it with the rest of your files.
Note: The “fit” function in keras does not accept a dataframe and only takes a matrix. If you
want to pass a dataframe as training or validation data to the fit function, you must first use
“as.matrix” function to convert it to matrix before passing it to the fit function; for example,
“as.matrix(your training dataframe)” or “as.matrix(your validation dataframe)”.
# 12. (0.4 points) Measure the performance of your best model (after tuning) on the test set and
compute its RMSE. Note that you must reverse the square root transformation you did in q3
by taking the square of the predictions returned by the neural network model and compare it to
the original count value (taking the square of the test target values). Doing this helps us get the
RMSE in the original scale.
# 13. (0.6 points) Use a simple linear regression model to predict the count. Train and test your
model on the same data you used to train and test your best neural network model. Compare
the RMSE of the linear model on the test data with the RMSE of the neural network model.
How does your neural network model compare to a simple linear model? Note that you need to
reverse the square root transformation again in this case like what you did in q12.
The submission must be in these formats
• A html file; You run all the code cells, get all the intermediate results and formalize your
answers/analysis, then you click ”preview” and this will create a html file in the same directory as
your notebook. You must submit this html file or your submission will not be graded.
Please note that if you knit your R notebook once, then you will lose the ”preview”
button! So do not knit!
