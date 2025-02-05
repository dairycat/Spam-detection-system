# Spam-detection-system
Text Data Processing: This mainly involves the public dataset we obtained from Kaggle. We divided the data into a test set (800 emails, including 160 spam emails) and a training set (200 emails, including 40 spam emails). The ratio of normal emails to spam emails is 4:1.

Dictionary Creation: We traverse the spam email text and perform text analysis. The main task in this module is to analyze the frequency of each word and count the frequency of each word.

Feature Extraction: This module focuses on extracting the optimal feature matrix from each email in the training set. The feature matrix for each email includes the frequency of the most frequent words (determined during the optimization of the feature matrix size).

Determining the Optimal Size of the Feature Matrix: We set the feature matrix dimension to range from 1000 to 2000. We want to find the dimension that maximizes the F1 score for the algorithm. Therefore, we test the effect of different dimensions from 1000 to 2000 with a step size of 100, aiming to find the dimension that results in the optimal F1 score. The optimal F1 score is the average F1 score of the six algorithms under their best parameters (also determined through optimization). (We attempted to use the eight most common algorithms, but the Random Forest and Gradient Boosting algorithms are random algorithms, and even after multiple iterations and averaging, we could not stabilize the optimal size, so we only use the six non-random algorithms).

Basic Algorithm Module: This module uses the eight most common classification algorithms: Naive Bayes, Decision Tree, KNN, SVC, Logistic Regression, Random Forest, AdaBoost, and Gradient Boosting for classification. In the testing code, we output precision, recall, and F1 score, using the weighted average of precision and recall as the metric to evaluate algorithm performance.

Multi-Algorithm Combination Module: We use multiple algorithms to predict results, applying logical operations such as AND and OR. AND means that a mail is classified as spam only when both algorithms classify it as spam. OR means that a mail is classified as spam if either of the algorithms classifies it as spam. We also propose a method to find the optimal weight for each algorithm, which maximizes the F1 score.

Detailed Design:
The detailed design for the six modules is as follows:

Text Data Processing: This includes the public dataset we obtained from Kaggle. We divided it into a test set (800 emails, with 160 spam emails) and a training set (200 emails, with 40 spam emails). The ratio of normal to spam emails is 4:1.

Dictionary Creation: Since the emails are in English, we split the text into individual English words to construct an array. We primarily use the split() function for this, iterating through the words to count the frequency of each word across all the emails. We also removed high-frequency but meaningless short words (1 or 2 letters) and long meaningless words (more than 16 letters).

Feature Extraction: After the previous steps, each email is transformed into a list of (word, frequency) pairs. We calculate the optimal number of features (ai) to determine the feature matrix size, which will improve the F1 score of the algorithm.

Determining the Optimal Size of the Feature Matrix: We start with a feature matrix size of ai = 1000 and iterate with a step size of 100, up to 2000. For each size, we optimize the algorithm's parameters and evaluate the best value that maximizes the F1 score. After completing the iterations, the code outputs the optimal feature matrix size (i.e., when the word's frequency rank is within the top ai, it is selected as a feature word), and it also outputs the optimal values for some algorithms that require parameter adjustment.

Basic Algorithm Module:

Naive Bayes Algorithm: In sklearn, there are five types: BernoulliNB, GaussianNB, MultinomialNB, ComplementNB, and CategoricalNB. MultinomialNB is suitable for discrete features that follow a multinomial distribution, such as word frequencies in text classification, making it the most appropriate for spam email classification. It uses prior probabilities to compute posterior probabilities. We found that alpha=1.0e-10 gives the best F1 score, indicating that the standard maximum likelihood estimation is appropriate for our dataset (when alpha=0, a warning appears, so we set it to alpha=1.0e-10).

Decision Tree Algorithm: In classification problems, a decision tree represents a classification process based on features. We use tree.DecisionTreeClassifier() in sklearn. The tree identifies the best node and branch method by minimizing impurity, using either information entropy ("entropy") or Gini index ("gini"). We found that using information entropy resulted in a higher F1 score.

KNN Algorithm: KNN is a basic classification method where the label of a new instance is determined by the majority vote of its k nearest neighbors. In sklearn, we use KNeighborsClassifier() for modeling. We found that using the default uniform weights resulted in the best F1 score. The n_neighbors parameter is optimized by iteration.

SVC Algorithm: Support Vector Classifier (SVC) is used in sklearn with SVC(). After tuning, we found that using the linear kernel gave the best performance due to its simplicity and computational efficiency. The regularization parameter C is optimized by iteration.

Logistic Regression Algorithm: Logistic regression builds a cost function for regression or classification problems and iteratively optimizes it to find the best model parameters. In sklearn, we use LogisticRegression(). After tuning, we found that the penalty term l2 and the solver lbfgs, with a tolerance of 0.0001, provided the best performance. The regularization coefficient C is optimized by iteration.

Random Forest Algorithm: A Random Forest is a collection of uncorrelated decision trees. In sklearn, we use RandomForestClassifier() to model it. After tuning, we found that using n_estimators=200 and criterion='entropy' yielded the best results.

AdaBoost Algorithm: AdaBoost combines multiple weak classifiers into a strong one. In sklearn, we use AdaBoostClassifier(). The number of estimators n_estimators and the learning rate learning_rate are optimized by iteration.

Gradient Boosting Algorithm: Gradient Boosting uses decision trees as base functions for boosting. In sklearn, we use GradientBoostingClassifier(). We optimize the number of estimators n_estimators and the learning rate learning_rate through iteration.

Multi-Algorithm Combination Module: We use the results of multiple algorithms in combination to predict the outcome:

(1) If both algorithms classify the email as spam, it is considered spam.
(2) If at least one of the two algorithms classifies the email as spam, it is considered spam.
(3) If two out of three algorithms classify the email as spam, it is considered spam.
(4) If three out of four algorithms classify the email as spam, it is considered spam.
We also solve for the optimal weight of each algorithm that maximizes the F1 score, with weights summing to 1 and a minimum weight of 0.05, with a step size of 0.1.
During the experiments, we observed that when using multiple algorithms, combining their results often improved or approximated the optimal performance.
