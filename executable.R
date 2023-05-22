library(data.table)
library(Rtsne)
library(ggplot2)

digits <- read.table("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/digits/digits.txt")
labels <- read.table("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/digits/labels.txt")

centered_digits <- sweep(digits, 2, colMeans(digits), "-")

digits <- as.data.table(digits)
labels <- as.data.table(labels)
digits[, label := labels[, V1]]
digits[, label := labels[, V1]]

sort(unique(digits[, label]))

set.seed(123) # for reproducibility

data_train <- digits[0L]
data_test <- digits[0L]
for (i in 0:9){
  digits_sub = digits[label == i]
  nrow_sub = round(nrow(digits_sub)/2)
  train_rows = sample(nrow(digits_sub), size = nrow_sub)
  
  
  train_table <- digits_sub[train_rows, ]
  
  test_rows <- setdiff(1:nrow(digits_sub), train_rows)
  test_table <- digits_sub[test_rows, ]
  
  data_train <- rbind(data_train, train_table)
  data_test <- rbind(data_test, test_table)
}

labels_train <- data_train[, .(label)]
labels_test <- data_test[, .(label)]

data_train[, label := NULL]
data_test[, label := NULL]


# write these data to csv to read from python.
fwrite(data_train, "C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/train_data.csv")
fwrite(data_test, "C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/test_data.csv")
fwrite(labels_train, "C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/train_labels.csv")
fwrite(labels_test, "C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/test_labels.csv")



## Answer 3 

# We are going to apply t-SNE with default values and then we are going to change parameters 
# and observe their effects.

 
set.seed(123)

my_digits <- as.data.table(digits)

tsne <- Rtsne(my_digits[, -c("label")], perplexity = 30, max_iter = 1000, eta=200)

tsne_dt <- data.table(tsne_x = tsne$Y[, 1], tsne_y = tsne$Y[, 2])
my_digits[, label := as.factor(label)]

my_palette <- colorRampPalette(colors = c("red", "green", "blue", "orange", 
                                               "purple", "yellow", "cyan", 
                                               "magenta", "brown", "pink"))(10)
                                               

ggplot(tsne_dt, aes(x= tsne_x, y= tsne_y, color=my_digits[,label])) + geom_point(alpha=0.5) + 
  scale_color_manual(values = my_palette)+
  labs(x = "t-SNE X", y = "t-SNE Y", color = "Labels")
 

# The 2-D graph shows that label 5 has not been clustered properly. 
# Except label 5, the labels seem to be clustered properly.
# 
# Now I am going to change parameters and observe the effects on the reduction. 
# Start with perplexity.

 
set.seed(123)

my_digits <- as.data.table(digits)

tsne <- Rtsne(my_digits[, -c("label")], perplexity = 5, max_iter = 1000, 
              eta=200)

tsne_dt <- data.table(tsne_x = tsne$Y[, 1], tsne_y = tsne$Y[, 2])
my_digits[, label := as.factor(label)]

my_palette <- 
  colorRampPalette(colors = c("red", "green", "blue", "orange", 
                                               "purple", "yellow", 
                                   "cyan", "magenta", "brown", "pink"))(10)
                                               

ggplot(tsne_dt, aes(x= tsne_x, y= tsne_y, color=my_digits[,label])) + 
  geom_point(alpha=0.5) + 
  scale_color_manual(values = my_palette)+
  labs(x = "t-SNE X", y = "t-SNE Y", color = "Labels")
 

# When we lower **perplexity** to 5, the t-SNE detects narrower clusters compared to perplexity being equal to 50.
# Also, the graph is now less dense.
# This is an expected result.
# 
# **Learning rate** is represented by eta in R. 
# Let's increase learning rate to 1000 from 200 and observe the graph.

 
set.seed(123)

my_digits <- as.data.table(digits)

tsne <- Rtsne(my_digits[, -c("label")], perplexity = 50, max_iter = 1000, eta=1000)

tsne_dt <- data.table(tsne_x = tsne$Y[, 1], tsne_y = tsne$Y[, 2])
my_digits[, label := as.factor(label)]

my_palette <- colorRampPalette(colors = c("red", "green", "blue", "orange", 
            "purple", "yellow", "cyan", "magenta", "brown", "pink"))(10)


ggplot(tsne_dt, aes(x= tsne_x, y= tsne_y, color=my_digits[,label])) + geom_point(alpha=0.5) + 
  scale_color_manual(values = my_palette)+
  labs(x = "t-SNE X", y = "t-SNE Y", color = "Labels")
 

# When learning rate is increased, the algorithm works quite faster. The graph does not change drastically from the
# default one.


 
set.seed(123)

my_digits <- as.data.table(digits)

tsne <- Rtsne(my_digits[, -c("label")], perplexity = 50, max_iter = 1000, eta=50)

tsne_dt <- data.table(tsne_x = tsne$Y[, 1], tsne_y = tsne$Y[, 2])
my_digits[, label := as.factor(label)]

my_palette <- colorRampPalette(colors = c("red", "green", "blue", "orange", 
            "purple", "yellow", "cyan", "magenta", "brown", "pink"))(10)


ggplot(tsne_dt, aes(x= tsne_x, y= tsne_y, color=my_digits[,label])) + geom_point(alpha=0.5) + 
  scale_color_manual(values = my_palette)+
  labs(x = "t-SNE X", y = "t-SNE Y", color = "Labels")
 

# With lower learning rate, clusters are a little bit wider.
# 
# Now i am going to observe the effect of **number of iterations**.


set.seed(123)

my_digits <- as.data.table(digits)

tsne <- Rtsne(my_digits[, -c("label")], perplexity = 50, max_iter = 2000, eta=50)

tsne_dt <- data.table(tsne_x = tsne$Y[, 1], tsne_y = tsne$Y[, 2])
my_digits[, label := as.factor(label)]

my_palette <- colorRampPalette(colors = c("red", "green", "blue", "orange", 
            "purple", "yellow", "cyan", "magenta", "brown", "pink"))(10)


ggplot(tsne_dt, aes(x= tsne_x, y= tsne_y, color=my_digits[,label])) + geom_point(alpha=0.5) + 
  scale_color_manual(values = my_palette)+
  labs(x = "t-SNE X", y = "t-SNE Y", color = "Labels")


# When we decrease number of iterations, the clustering becomes less accurate.