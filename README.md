# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH ID SOLUTIONS

*NAME*: ARYA GOSAVI

*INTERN ID*: CTO4DF2648

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

# DESCRIPTION : 
 A Decision Tree is a popular supervised machine learning algorithm used for both classification and regression problems. It mimics human decision-making by breaking down a complex decision process into a series of simpler decisions, creating a tree-like model of decisions and their possible consequences. Each internal node of the tree represents a test on an attribute (feature), each branch represents the outcome of the test, and each leaf node represents a class label (in classification) or a continuous value (in regression). The algorithm works by selecting the most significant attribute using measures like Gini impurity or information gain (based on entropy) and recursively splits the dataset into subsets until all or most of the data points in each subset belong to the same class.
 Using Scikit-learn, one of Python’s most widely used machine learning libraries, implementing a Decision Tree is both efficient and straightforward. The first step is to import the necessary libraries: pandas for handling dataframes, matplotlib.pyplot for plotting the decision tree, and Scikit-learn modules like DecisionTreeClassifier, plot_tree, and model evaluation tools such as train_test_split, accuracy_score, and classification_report. For demonstration purposes, the Iris dataset is often used due to its simplicity and well-structured nature. This dataset includes 150 samples of iris flowers, each described by four features—sepal length, sepal width, petal length, and petal width—and classified into one of three species: Setosa, Versicolor, or Virginica.
 Once the dataset is loaded using Scikit-learn’s load_iris() function, the features (X) and the target labels (y) are separated. The next step involves splitting the data into training and testing subsets using the train_test_split() function. This ensures that the model can be trained on one part of the data and evaluated on a separate, unseen portion, which helps in assessing the model’s generalization ability. Typically, a 70:30 or 80:20 train-test split is used. Then, a DecisionTreeClassifier object is created by specifying parameters such as criterion="entropy" (which uses information gain to decide splits) and a random_state for reproducibility. The model is trained using the .fit() method, which learns the splitting rules from the training data.
After training, predictions are made on the test set using the .predict() method. The performance of the model is then evaluated using accuracy_score, which gives the overall correctness of the model, and classification_report, which provides precision, recall, and F1-score for each class. These metrics are essential for understanding how well the model performs, especially in multi-class problems like the Iris dataset.
 One of the key strengths of decision trees is their interpretability. Scikit-learn provides a built-in plot_tree() function that allows you to visually inspect the structure of the tree. This visualization displays each decision rule, the number of samples at each node, class distribution, and the predicted class. Such transparency is invaluable for explaining model behavior to stakeholders or in regulated industries where interpretability is critical.
While Decision Trees are easy to implement and understand, they have some limitations. They are prone to overfitting, especially when the tree becomes too deep and captures noise in the training data. To combat this, one can prune the tree, limit its depth using the max_depth parameter, or use ensemble methods like Random Forests or Gradient Boosted Trees, which combine multiple decision trees to achieve better performance and generalization. 
