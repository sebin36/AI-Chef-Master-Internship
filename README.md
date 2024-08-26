1. Import Necessary Libraries
   
   Numpy - For performing numerical operations.
   Pandas - For data manipulation and preparation of model.
   Matplotlib & Seaborn - For Visualization.
   Sklearn, train_test_split - For spliting the data.
   Sklearn, NearestNeighbors - For Model creation.
   Sklearn, LabelEncoder - For encoding categorical values.
   Sklearn, accuracy_score, classification_report - For Model Performance Evaluation.
   Warnings Module - For preventing warnings.


2. Reading the File

   Using the 'pd.read_csv' function, the file is opened.
   Column names are assigned to each columns in the dataset.
   Encoding specifies the character encoding used for decoding the file.
   The first 10 rows of the data is viewed using 'data.head(10)' function.


4. Data Preprocessing

   Performs basic preprocessing on each dataset(movies, users and ratings).
   Column Conversion - Converts each genre in the genre column(movies dataset) into a list by spliting each genre based on the '|' symbol.
   Other tasks include checking for dataset size, checking for null values, etc.


 5. Merging the Data.

    Merging of the three datasets is required inorder to use the dataset efficiently for model building.
    For that, the 'ratings' dataset is merged with 'movies' dataset based on a common attribute 'movieid' and is stored in a variable named 'ratings_movies'.
    The 'ratings_movies' is again merged with 'users' dataset based on common attribute 'userid' and stored in a variable 'data'.


6. Label Encoding

   Converting categorical data into numbers allows the model to process and analyze the data effectively.
   Here, the gender column is converted into numerical values, therefore the value 'M' will be converted into 0 and 'F' will be converted into 1.


7. Drop Unnecessary Columns

   Now, we drop the columns that are not used for model building such as 'timestamp' and 'zipcode'.


8. Preparing the data for collaborative filtering

   The main columns used for this function are 'userid', 'movieid' and 'ratings'.
   pivot - It is a function in Pandas used to reshape data, turning unique values from one column into separate columns and another column's values into the   
   DataFrame's rows. The purpose is to create a matrix where each row represents a user, each column represents a movie, and the values in the matrix are the
   ratings given by the users to the movies.

   fillna - After the pivot operation, there might be many missing values (NaN) in the matrix, which occur when a user has not rated a particular movie. The       
   'fillna(0)' method is used to replace all the NaN values with 0.

   The final Output will be:
   Rows represent users ('userid').
   Columns represent movies ('movieid').
   The values in the matrix represent ratings, with 0 indicating no rating.

   The matrix is then stored into a variable named 'user_item_matrix'.


9. Splitting the data

   The matrix is then transferred into the 'train_test_split' function for splitting the data into X_train and X_test.
   The value for Test size is 0.25.
   Random state is set as 42.


10. Calling the Model

    Here, we are using the K - Nearest Neighbors algorithm since it is a commonly used algorithm for recommendation systems. It's commonly used for finding the     
    closest points (neighbors) in a dataset.
    
    Metric - The metric parameter specifies the distance metric to use for finding the nearest neighbors.
    Cosine - Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. In the context of a recommendation 
    system, cosine similarity is used to determine the similarity between users or items based on their ratings.

    Algorithm - The algorithm parameter specifies the algorithm used to compute the nearest neighbors.
    Brute - Brute force search is a straightforward method that compares each point in the dataset to every other point to find neighbors.


12. Fitting the Model

    The model needs to be trained on training data inorder to give recommendations. For that we use 'fit' function.
    
    fit(X_train) - This method trains the KNN model on the provided training data (X_train).


13. Movie Recommendation Function

    This function takes values such as user_id, model, user_item_matrix and a 'n' value(number of nearest neighbors) to make user based recommendations.
    
    The function can be divided into 2 parts:

    Distance Calculation - The distance calculation in the 'recommend_movies_knn' function is used to find and measure how similar other users (or items) are to 
    the target user. The smaller the distance, the more similar they are. This helps identify the most similar users or items, which are then used to make 
    recommendations.

    Recommendations - The recommendations section in the 'recommend_movies_knn' function generates a list of movie suggestions for the target user. After finding 
    the most similar users, the function picks movies that these similar users liked and recommends them to the target user, assuming they will have               
    similar tastes.


14. Making Recommendations

    Using the 'recommend_movie_knn' function, the model will give 10 user recommendations based on the user_id passed to the model as test sample. The model will 
    give recommendations who have similar interest that of given test user_id.


15. Model Evaluation

    For Model Evaluation, we use Mean Squared Error(MSE). MSE is a common metric for evaluating the accuracy of regression models by measuring the average squared     difference between predicted and actual values.

    After MSE Calculation we got a value of 1.08 which suggests that the model is slightly prone to prediction errors and need to be improved.


16. Feedback Loop

    Feedback Loop updates the dataset by adding new values into the dataset and retraining it thereby increasing the chance of more accurate recommendations.
