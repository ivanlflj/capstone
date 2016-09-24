# Capstone Project
## Machine Learning Engineer Nanodegree
Ivan Landim Frota Leitão Junior  
September XXth, 2016

## I. Definition
The stock market is full of opportunities, every day billions of reais are negociated in the market. Over than half a million citizens have money invested in Bovespa with 120 billion reais of assets inside Bovespa, only in the hand of citizens, excluding companies owning assets. The amount of money and opportunity are big but also are the risks, plenty of people lose their money on the market, some of them losing large amounts.

In this scenario, our objective is to create a machine learning algorithm that would be able to predict the day a stock will go up. With this in hand, we would be able to increase the probability of ours intraday trades. The idea is to improve the trades that are done in the stock for just one day. This opens the possibility to invest with leverage (lended money) for free, only considering the risk of the operation. (In this report leverage will not be explained, check [this link](http://www.investopedia.com/articles/investing/073113/leverage-what-it-and-how-it-works.asp) for more information.)

The algorithm name is PUM (Predictive Upside of the Market).

### Project Overview
PUM will predict a day that the stock price will move up. The objective here is not to predict the amount it will go up but the day it will happen. Other studies can try to predicte the better moment in these days to enter in the market or to predict the percentage it will move up, using this study as a starting point.

The choosen stock to be analyzed in the study is the ITSA4 (ITAUSA INVESTIMENTOS ITAU S.A. - Preferred stock). The reason for choosing this stock is because it has a long historical data and the negotiation is substancial. Other stocks can be analysed later using PUM just fitting their data.

The data used is raw data of the negotiations from 2006 to 2015, it will be over 2000 days of negotiation and it will be gathered from the [Bovespa website](http://www.bmfbovespa.com.br/). The data gives the price variation of the stocks in each day (open, high, low, close) and volume of the negotiation in the day.

As a bonus analysis, we will verify if buying in the open price and selling in the low price will give money if we follow PUM prediction. It is possible that the losses in the days it misses are bigger than the gains in the days it is right, this is the main reason to have it as a bonus analysis.

### Problem Statement
We invest in a market that moves up and down, the probability that a given day it moves up is lower than 50%, in the last 10 years. Our intention is to have a higher probability in the days we decide to invest.

We will break the main objective into steps:
- Gather and prepare the data for analysis
- Include [technical analysis](http://www.investopedia.com/university/technical/) indicators
- Clean the data for usage in the Machine Learning
- Divide the data in training and test data
- Prepare the machine learning with Decision Tree, K-neighbors, AdaBoost and Random Forest
- Test is with and without PCA
- Evaluate the performance and pick the best model
- (Bonus analysis) Check the stock performance of the choosen model in the test data

### Metrics
To evaluate the performance of the model we have decided to use the precision score. The main objective of our analysis is to be in the market when there is a higher probability of gaining money. There is no problem that the market goes up when we are outside of it. We want to avoid the market going down when we are inside it.

For this reason, precision score was the best score in our point of view. We try to maximize the amount of true positive (positive trades that we considered positives) and to reduce the false positives (negative trades that we considered positive).


## II. Analysis

### Data Exploration
The data used in this project is the historical prices and volumes of the stock for 10 years. The close price of the stock is in the graph above.

![Prices graph](https://github.com/ivanlflj/capstone/blob/master/prices.png)

To improve the data we have added some new features using the technical analysis indicators as Moving Average and Relative Strenght Index, in the images below. The detailed information about them are in the [Jupyter notebook](https://github.com/ivanlflj/capstone/blob/master/stock_analysis_code.ipynb).

![Moving Average](https://github.com/ivanlflj/capstone/blob/master/ma50.png)

![Relative Strenght Index](https://github.com/ivanlflj/capstone/blob/master/rsi.png)

The volume of trades in each day are also presented here.

![Volumes](https://github.com/ivanlflj/capstone/blob/master/volumes.png)

Verifying the amount of days it moved up, we can see that it is less than half of them.

![Days up](https://github.com/ivanlflj/capstone/blob/master/days_up.png)

We also analized it depending of the day of the week and the month of the year.

![Up week](https://github.com/ivanlflj/capstone/blob/master/up_week.png)

![Up month](https://github.com/ivanlflj/capstone/blob/master/up_month.png)

### Exploratory Visualization
We also had performed tests checking if moving up in the previous 5 days changed the probability of moving up today.

![Up previous](https://github.com/ivanlflj/capstone/blob/master/up_previous.png)

From the volumes we could analyze the last volumes in relation of the previous volumes.

![Up volumes](https://github.com/ivanlflj/capstone/blob/master/up_volumes.png)

We also verified it for different Moving averages (in relation of the last price) and Relative Strenght Index.

![MA5 up](https://github.com/ivanlflj/capstone/blob/master/ma5up.png)

![MA8 up](https://github.com/ivanlflj/capstone/blob/master/ma8up.png)

![MA21 up](https://github.com/ivanlflj/capstone/blob/master/ma21up.png)

![MA50 up](https://github.com/ivanlflj/capstone/blob/master/ma50up.png)

![RSI2 up](https://github.com/ivanlflj/capstone/blob/master/rsi2.png)

![RSI5 up](https://github.com/ivanlflj/capstone/blob/master/rsi5.png)

![RSI7 up](https://github.com/ivanlflj/capstone/blob/master/rsi7.png)

![RSI14 up](https://github.com/ivanlflj/capstone/blob/master/rsi14.png)

### Algorithms and Techniques
Four algorithms were used to model the data: K-Nearest Neighbors, Decision tree, AdaBoost and Random Forest. For all of them we used some the GridSearch method to improve the parameters. There is a brief presentation of each algorithm below:

K-Nearest Neighbors:
The model is based on check the a certain number (K) of nearest neighbors of the point we are trying to predict.
![K-Nearest Neighbors example](http://scikit-learn.org/stable/_images/plot_classification_0011.png)

Decision tree:
The decision tree creates several branches and conditions to predict the outcome of a certain model.
![Decision tree example](http://scikit-learn.org/stable/_images/iris.svg)

AdaBoost:
The core principle of AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. 

![AdaBoost example](http://scikit-learn.org/stable/_images/plot_adaboost_hastie_10_2_001.png)

Random Forest:
The random forest is a perturb-and-combine techniques specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.

![Comparison with Random Forest](http://scikit-learn.org/stable/_images/plot_forest_iris_0011.png)

### Benchmark
We can use two ways to predict a good model.

First, we need to be better than random. The chance of chosing a day that the stock goes up is slightly less than 50%. Any improvement in this performance is positive.

Second, if we buy stocks in the openning of the day and sell in the close we will have positive results. As mentioned before we do not expect to use the model to buy and sell stocks purely. The idea is to use a intraday model in combination with this one but anyway it is a good test.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
From the original data, the days of each negociation were transformed in days of the week.

The past prices became MAs, RSIs and prices for the last 15 days. All the prices were also divided by the last price, so it became a function of the last price, becoming numbers around 1.

The past volumes became the last 5 volumes also divided by the last volume, so we have number around 1.

We also used a function that selected the K best features to send to the data, in this case we changed K between 30, 50 and 70.

PCA and ICA were tested but none of them delivered a good result. Some of the Ensemble models performed better with them but it was only because of the bias that the model had. In the end we considered the best model not having it.

### Implementation
As previously mentioned the code is documented in the [Jupyter notebook](https://github.com/ivanlflj/capstone/blob/master/stock_analysis_code.ipynb) with explanation but here we will mention the most important things.

The prediction of the model was really bad until we changed the target function. As mentioned in the Metrics chapter the objective of the model is to improve the chance of the market to move up and not to be always right. Since the models try to be always right we had to change a little the target. The target become to know when the stock would move 1% up. Using this target we were able to increase the chance of predicting a up in the price.

To separate the data between train and test data we had to use the 1% up in the stratified data, in this case we had a good distribution of the days that moved more than 1% up.

For all the algorithms we defined a GridSearchCV with the default crossvalidation of 3-fold. Since the data was already shuffled from the train_test_split we didn't need to worry to shuffle it.

In the beginning we also tested the SVC but it was taking too long to run and it became impossible to use.

It worth mentioning that we used Pipelines and it can be seen in the [Jupyter notebook](https://github.com/ivanlflj/capstone/blob/master/stock_analysis_code.ipynb).

### Refinement
All change in data done to improve the model were already mentioned previously.

The Gridsearch was used since the beginning in all models so there is not much refinement here.

The Pipelines were implemented later to be able to run the code faster and in a way that run all the possibilities. Before the pipelines the performance of the model were similar to random but being able to run more models faster it is possible to get several results and get the best one.

## IV. Results

### Model Evaluation and Validation
The chosen model is the Decision Tree not using PCA or ICA.

Adaboost and Random Forest gave better performances in the training and testing data but they have really low trades, so it is probable overfitted.

In the test data around 600 days, the decision tree has 26 trades in comparison of the ensemble models that had around of 5 trades. Decision tree trades 4% of the days what is low but it is a lot better than the ensemble models.

In the training data the decision tree was able to reach a perfomance of 82%.

In the test data the decision tree had a 62% in the precision score.

The result appears to be a lot more reliable than the other models. We had the idea of preparing a learning curve and a validation curve but since we are training in the days that the market moves 1% up and checking with the days that the market moves up we were not able to perform it. We understand the importance of the curves but it would not be reliable here.

Imagening the learning curves it is possible that it would be clear that we are still suferring a little bit from variance and more data would improve our model. We can in the future prepare the model to be used in more stocks or get more years so the model improves the performance.

### Justification
In comparison with the benchmark the performance of the model is good.

By random the chance of picking a day that would move up is less than 50% and the model got a performance of 62%.

If we used our model to buy one stock in the opening of the market and sell it in the close we would get out of the market in this test data with a profit of 0.04. It is positive but it is not good and we can consider as 0. But as mentioned before the idea is to improve the performance of other intraday methods and not having a negative result is already good start to improve the other trades.

## V. Conclusion

### Free-Form Visualization
The main image to show the performance of the model is the total data prediction if it is a positive day or not. The image is shown below and it is the entire data with negative and positive days and the entire data only with the days expected to be positive.

![Result](https://github.com/ivanlflj/capstone/blob/master/result.png)

We can clearly see a diference in the probabilities. In here it is showing the training data also what is not perfect but it is a good visualization.

### Reflection
The capstone was really good to do. It strached my understanding about Machine Learning and it showed me it is not easy. Getting a real world problem brings a totally new challenge. Using pipelines brings a total new possibility to test the models and improve the performance of the predictions.

The Machine Learning here was important to get a good solution. The project was harder than expected and the performance was a little below than the desired. Too little trades and/or low precision.

It is interesting to see how hard is to predict the market and its future.

We also tested the problem trying to run as a regression but we were not successful. I didn't test too much but it is one of the branches in the Github project.

### Improvement
As already mentioned the model could possibly be improve by including more stocks in the trading and getting more historical data.

With more data the ensemble methods and maybe the neural networks could be a possibility for the project. In the case of the neural networks I would need to study more the methodologies, I read a little thinking of implementing them but a have seen they are really useful in Big Data and we didn't have this much data, it could easily overfit as the ensembles.