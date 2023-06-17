train_data = pd.read_csv("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/train_data.csv", header=None, skiprows=1)
train_data = train_data.values
train_data = np.reshape(train_data, (2500, 400))

test_data = pd.read_csv("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/test_data.csv", header=None, skiprows=1)
test_data = test_data.values
test_data = np.reshape(test_data, (2500, 400))

train_labels = pd.read_csv("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/train_labels.csv", header=None, skiprows=1)
train_labels = train_labels.values
train_labels = np.reshape(train_labels, (2500,))

test_labels = pd.read_csv("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/data/test_labels.csv", header=None, skiprows=1)
test_labels = test_labels.values
test_labels = np.reshape(test_labels, (2500,))

### Answer 1.2

# As scikit learn already centers data, I have used non-centered data for the answer.


plt.clf()
pca = PCA(n_components=400)
pca.fit(train_data)

# order the eigenvalues
descending_eigenvalues = np.sort(pca.explained_variance_)[::-1]

plt.plot(range(1, 401), descending_eigenvalues)
plt.xlabel('Pincipal Components')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues of Principal Components')
plt.show()

### Answer 1.3


plt.clf()
# Display the sample mean as an image
train_mean = np.mean(train_data, axis=0)
image_sample_mean = np.reshape(train_mean, (20, 20))


plt.imshow(image_sample_mean, cmap='gray')
plt.title('Sample mean of the training data')
plt.axis('off')
plt.show()


# The shape of the sample mean has a curved shape. Considering most numbers are
#curvy, it is expected. It looks like 8.


plt.clf()
# display eigenvectors as an image
eigenvectors = pca.components_.reshape(-1, 20, 20)

chosen_eigenvectors = eigenvectors[:50]

fig, axes = plt.subplots(5, 10, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(chosen_eigenvectors[i], cmap='gray')
    ax.axis('off')

plt.suptitle('Chosen Eigenvectors')
plt.show()

### Answer 1.3

 {python, results='hide'}
subspace_dimensions = range(1, 201, 10)

train_error_pca = []
test_error_pca = []

for i in subspace_dimensions:
    # Perform PCA 
    pca = PCA(n_components=i)
    pca.fit(train_data)
    
    # Project the training and test data onto the subspace
    train_data_projected = pca.transform(train_data)
    test_data_projected = pca.transform(test_data)
    
    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(train_data_projected, train_labels)
    
    # Train the classifier
    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(train_data_projected, train_labels)
    
    # Predict labels
    train_predictions = classifier.predict(train_data_projected)
    test_predictions = classifier.predict(test_data_projected)
    
    # Calculate errors
    train_err = 1-accuracy_score(train_predictions, train_labels)
    test_err = 1-accuracy_score(test_predictions, test_labels)
    

    train_error_pca.append(train_err)
    test_error_pca.append(test_err)
 



## Question 1.4  

# Plot classification error vs. the number of components used for each subspace, and discuss
# your results. Compute the classification error for both the training set and the test set
# (training is always done using the training set), and provide two plots.

### Answer 1.4

 
plt.clf()
plt.plot(subspace_dimensions, train_error_pca, label='Training Error')
plt.plot(subspace_dimensions, test_error_pca, label='Test Error')
plt.xlabel('Number of Components')
plt.ylabel('Classification Error')
plt.title('PCA - Classification Error vs. Number of Components')
plt.legend()
plt.show()

 

# As seen when n_components is increased, the classification error is decreased continuously for training data, but
# it behaves differently for test data. For test data, the classification error is decreased until around number of components being equal to 23 
# and then it continuously increases. One possible reason for this situation is overfitting. The additional components may capture the noise in the training data and that's why, it would be less effective in an unseen test data.

# QUESTION 2 [35 points] 

# In this question, you will use Isomap (J. B. Tenenbaum, V. de Silva, J. C. Langford, A Global
# Geometric Framework for Nonlinear Dimensionality Reduction,". Science, vol. 290, pp:2319-
# 2323, 2000) to map the 400-dimensional data onto lower dimensional manifolds.

## Question 2.1

# Use Isomap to obtain low-dimensional embeddings of the digits data. Note that you
# need to use the full data set, i.e., 5,000 patterns, but you may still have several patterns
# that were not embedded. This is a common observation in many techniques that are
# based on neighborhood graphs where the embedding implementation only uses the largest
# connected component of the neighborhood graph and ignores the patterns belonging to
# other components.

### Answer 2.1

# Apply Isomap for dimensionality reduction on the digits dataset.

 
digits = np.loadtxt("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/digits/digits.txt")
labels = np.loadtxt("C:/Users/yagiz/Desktop/4-2/GE-461/DimensionalityReduction/digits/labels.txt")
 

 
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2) # apply with 2 now

isomap.fit(digits)

embeddings = isomap.transform(digits)
 


## Question 2.2

#  Choose dimensions between 1 and 200 (choose at least 20 different dimensions, the more
# the better) and train a Gaussian classifier for each dimensionality (do not forget to use
# half of the data for training and the remaining half for testing).

### Answer 2.2



subspace_dimensions = range(1, 201, 10)

train_error_isomap = []
test_error_isomap = []


for i in subspace_dimensions:
 
    isomap = Isomap(n_components=i)
    
    # Fit the Isomap model to the training data
    isomap.fit(train_data)
    
    # Transform the training and test data to obtain the low-dimensional embeddings
    train_data_embedded = isomap.transform(train_data)
    test_data_embedded = isomap.transform(test_data)
    
    
    # Train the classifier
    classifier = QuadraticDiscriminantAnalysis()
    classifier.fit(train_data_embedded, train_labels)
    
    # Predict labels
    train_predictions = classifier.predict(train_data_embedded)
    test_predictions = classifier.predict(test_data_embedded)
    
    # Calculate classification errors
    train_err = 1 - accuracy_score(train_predictions, train_labels)
    test_err = 1 - accuracy_score(test_predictions, test_labels)
    
    train_error_isomap.append(train_err)
    test_error_isomap.append(test_err)
 

## Question 2.3

# Plot classification error vs. dimension, and discuss your results. Compute the classification
# error for both the training set and the test set (training is always done using the training
# set), and provide two plots. The discussion should include the setting (particular choice
# for the parameters) for Isomap, the effect of dimensionality on the classification error, and
# comparison of the Isomap results with the PCA results.

### Answer 2.3

 
plt.clf()
plt.plot(subspace_dimensions, train_error_isomap, label='Training Error')
plt.plot(subspace_dimensions, test_error_isomap, label='Test Error')
plt.xlabel('Dimension')
plt.ylabel('Classification Error')
plt.title('Isomap - Classification Error vs. Dimension')
plt.legend()
plt.show()
 

# # As seen when n_components is increased, the classification error is 
# decreased continuously for training data, but
# # it behaves differently for test data. For test data, the classification
# error is decreased until around dimension is being equal to 23 and then it 
# starts to slowly increase. Again, it may be because of overfitting. With the 
# increase in the
# # dimension, the model may capture the noise in the train data and that makes 
# the model work poorly in an unseen data.

 
error_comparison = {'Train Error (PCA)': train_error_pca,
        'Train Error (Isomap)': train_error_isomap,
        'Test Error (PCA)': test_error_pca,
        'Test Error (Isomap)': test_error_isomap}

error_comparison = pd.DataFrame(error_comparison)

average_error = {'Mean Train Error (PCA)': np.mean(train_error_pca),
        'Mean Train Error (Isomap)': np.mean(train_error_isomap),
        'Mean Test Error (PCA)': np.mean(test_error_pca),
        'Mean Test Error (Isomap)': np.mean(test_error_isomap)}
average_error = pd.DataFrame(average_error)

print(f'Mean Train Error (PCA): {np.mean(train_error_pca)}\n Mean Train Error 
(Isomap): {np.mean(train_error_isomap)} \n Mean Test Error (PCA): 
  {np.mean(test_error_pca)} \n Mean Test Error (Isomap): 
    {np.mean(test_error_isomap)}')
 

# As seen PCA works better. The possible reason is that PCA outperforms Isomap 
#in capturing linear relationships. Probably there are linear relationships in 
# the dataset.

