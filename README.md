Restaurant Review Sentiment Analysis

This project performs sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

🧠 Project Description

The goal of this project is to classify restaurant reviews as positive or negative using machine learning. We use the Restaurant_Reviews.tsv dataset which contains 1,000 reviews along with sentiment labels (Liked = 1 for positive, 0 for negative).

The main steps include:

Text preprocessing and cleaning

Feature extraction using Bag of Words (BoW)

Model training using Gaussian Naive Bayes

Evaluation using accuracy and confusion matrix



---

📁 Files

Restaurant_Reviews.tsv: The dataset file in tab-separated format.

sentiment_analysis.py: The main script containing all implementation steps.



---

🛠️ Libraries Used

numpy
pandas
matplotlib
nltk
scikit-learn

Make sure to install them using:

pip install numpy pandas matplotlib nltk scikit-learn


---

🚀 How to Run

1. Clone the repository or download the script and dataset.


2. Make sure Restaurant_Reviews.tsv is in the same directory as the script.


3. Run the script:

python sentiment_analysis.py




---

🧹 NLP Preprocessing Steps

Each review goes through the following steps:

Removing non-letter characters using RegEx

Converting to lowercase

Tokenization (splitting into words)

Removing stopwords (excluding "not")

Stemming using PorterStemmer

Rejoining the cleaned words into a single string



---

🧰 Machine Learning Workflow

1. Vectorization: Converts cleaned reviews into numerical vectors using CountVectorizer with a max of 1500 features.


2. Train/Test Split: 80% training, 20% testing using train_test_split.


3. Training: A Gaussian Naive Bayes model is trained.


4. Prediction & Evaluation:

Generates predictions on test data

Computes confusion matrix

Calculates accuracy score





---

📈 Sample Output

Example prediction comparison:

[[1 1]
 [0 0]
 [1 1]
 [0 1]]

Confusion matrix:

[[55 42]
 [12 91]]

Accuracy score:

0.73


---

✅ Requirements

Python 3.x

NLTK corpus stopwords (automatically downloaded in code)



---

🔍 Future Improvements

Try other models (SVM, Logistic Regression, etc.)

Tune hyperparameters

Use TF-IDF instead of Bag of Words

Include visualization of results



---

📬 Contact

For any questions or suggestions, feel free to reach out.
