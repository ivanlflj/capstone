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
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
