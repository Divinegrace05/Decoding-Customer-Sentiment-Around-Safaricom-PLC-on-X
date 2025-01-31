# Decoding Customer Sentiment Towards Safaricom PLC on X
![SentimentAnalysis](https://github.com/user-attachments/assets/cab24d9c-3e86-4a5e-bcb5-b17b010cafb2)


## Project Summary
In today’s digital era, social media plays a pivotal role in shaping public sentiment, particularly in the financial domain. This study focuses on analyzing social media discussions, specifically tweets discussing Safaricom PLC on X (formerly Twitter), leveraging Natural Language Processing (NLP) techniques. By systematically collecting and analyzing tweets, this project uncovers insights into the prevailing sentiment surrounding Safaricom PLC, which can significantly influence consumer perceptions and investment decisions. Sentiments are categorized into positive, negative, and neutral classes, enabling a comprehensive understanding of public opinion and its implications for brand reputation and market performance.

## Business Problem
Safaricom PLC faces significant challenges in customer retention and brand reputation management in a competitive telecommunications market. Negative sentiment expressed on social media can indicate dissatisfaction with services or products, potentially leading to customer churn. This project utilizes sentiment analysis to gain insights into customer feedback, identify recurring issues, and understand the factors driving consumer sentiment. These insights empower Safaricom to enhance service delivery, tailor marketing strategies, and improve customer support, fostering customer loyalty and strengthening its brand image.

## Installation
To run this project, ensure you have Python installed and install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud emoji
```

## Dataset
The dataset consists of tweets discussing Safaricom PLC collected from X. Ensure the dataset is named `safaricom_tweets.csv` and placed in the project directory. It includes variables such as:
- `Text`: The tweet content.
- `Language`: Language of the tweet.
- `Favorites`: Number of likes.
- `Retweets`: Number of retweets.

## Exploratory Data Analysis (EDA)
### Key Steps:
1. Dataset shape, structure, and missing values analysis.
2. Language distribution and frequency of unique tweets.
3. Relationship between favorites and retweets through scatter plots.

### Outputs:
- Bar plot of tweet language distribution.
- Scatter plot illustrating the relationship between favorites and retweets.

## Data Cleaning and Preprocessing
### Key Steps:
1. Lowercasing text.
2. Removing URLs, mentions, hashtags, and emojis.
3. Tokenization, stopword removal, and lemmatization using NLTK.
4. Adding a `Preprocessed_Text` column with the cleaned tweets.

### Example:
- Original: "@Safaricom_ke Please fix my internet! 😡 #safaricom"
- Preprocessed: "please fix internet"

## Sentiment Analysis
Using NLTK's Sentiment Intensity Analyzer (VADER):
1. Assign sentiment scores to preprocessed text.
2. Categorize tweets as Positive, Negative, or Neutral based on sentiment scores.

### Visualization:
- Sentiment distribution bar chart.
- Word clouds for positive, negative, and neutral sentiments.

## Model Training and Evaluation
### Models Used:
1. Logistic Regression
2. Random Forest
3. Naive Bayes
4. Support Vector Classifier (SVC)

### Process:
1. Convert text into numerical representation using TF-IDF Vectorizer.
2. Split data into training and test sets.
3. Train models and evaluate using classification reports and confusion matrices.

### Key Metrics:
- Precision, Recall, F1-score.
- Confusion Matrix for visualization.

### Hyperparameter Tuning:
- Logistic Regression optimized using GridSearchCV.
- Best parameters and performance score included.

## Visualizations
1. Feature importance for Random Forest.
2. Top 10 important features for sentiment classification.
3. Confusion matrices for each model.

## Results
- Comprehensive comparison of model performance.
- Logistic Regression tuned for optimal results, achieving the highest accuracy.

## Conclusion
This analysis highlights customer sentiment towards Safaricom PLC, enabling actionable insights for improved customer satisfaction and brand loyalty. The best-performing model provides reliable predictions for sentiment classification, supporting Safaricom's efforts to enhance its services.

## Future Work
- Incorporate more social media platforms for broader sentiment analysis.
- Perform topic modeling to identify key themes in customer feedback.
- Use deep learning techniques for improved accuracy.

## Acknowledgments
Special thanks to Safaricom PLC for inspiration and to the open-source community for the tools used in this project.

