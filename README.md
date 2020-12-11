# SupersizeMe
#### Marshal Taylor
#### Supporting Code found in SuperSizeMe.ipynb

## Background
The original idea for the project came to me during my experiences backpacking. I've eaten at plenty of Ma and Pa restaurants after a long week of hiking. Usually, These restaurants are not required to post their nutritional data for the food that they serve, due to their small business size. I originally planned on building a model that would take categorical attributes about the food (such as the type of food, serving size, and comparable restaurants), and return the predicted nutritional data. This approach wouldn't be very accurate given the limited number of features, and the many targets that I hoped to predict. In the end, I decided that simply looking up a similar food item for it's nutritional data online would've been the simpler and more logical approach. The project then pivoted. In this new model, I would take all of the nutritional data about a food item, and predict what type/category a food item belonged to.

The purpose of this model is to explore the similarities and differences between the foods we get at restaurants. If we supply a machine learning model with all of the nutritional facts of a piece of food, as well as the restaurant that the item came from, can it correctly predict what category of food that we purchased (ie, Hamburger, shake, fries, etc).
A model like this could be helpful for a researcher who would like to run a data analysis on a set composed of uncategorized nutritional data. Our model could go through and categorize the food in the dataset automatically for them.

## Data
### Collection and Cleaning
Since the inspiration of the project came from my backpacking adventures, and my love for a good burger, I chose 3 restaurants that encapsulated most types of burgers that I've had before:
  * McDonalds
  * In N Out
  * Red Robin
  
Since large chains of restaurants are required to post their nutritional information in a standardized way, it meant the data collection was fairly easy. The data from McDonalds came from a kaggle dataset that I simply downloaded. The data for In N Out and Red Robin both came off PDF's from their respective websites. I've never scraped data off of a PDF before, and I thought it would be a fun challenge to overcome. It turned out to be much more difficult than I first expected.

To scrape the data off of the PDF's, I used the function read_pdf from the package tabula. Tabula's read_pdf creates a list of dataframes. These data frames usually don't match dimensions, are missing data, and include overlapping data from other data frames. This meant the cleaning process was rather tedious and difficult, especially for the Red Robin dataset.

Of the largest issues I found was Tabula's tendency to combine rows of data into the same column. To combat this, I would shift data over in a dataframe to make space for the column that was combined. I would then need to split the contents of the column into 2, and then convert the columns into their correct data type.

### Creation
Our Red Robin data did not include daily percentage points for each of the different nutritional attributes. I used a KNN imputer to find these missing values. I chose this imputation method since the KNN model searches for similar rows in the dataset to choose their values. Since every restaurant serves similar items of food, the KNN produced accurate results.

In addition to the imputation of some data, I added a new categorical variable that labeled which restaurant each of the items of food originated from.

The data scraped from the PDF's did not include the category of food each item belonged to. I built a function that would search for certain strings in the name of the item to classify which category it belongs to. 

### Exploratory Data Analysis
Since the whole project is exploratory in nature, the actual EDA in the project is small. I used a pairplot to see how each of the features correlated with every other feature. The most interesting thing that can be gleaned from the graphs is the positive correlation that nearly every feature has with every other feature. Some of the correlations are weak, but most are still positive.

This makes sense. As a food item gets larger it usually has more of everything, every nutritional feature.

The only place where this relationship isn't true is the relationship between sugar and sodium. For both of these features, there are "two groupings'' of data. The first group of data has an extremely positive slope, shooting off rapidly from the other points of data. The second group of data has a much less positive slope. A more interpretable way to approach this relationship, is to say that items have a lot of sugar, a lot of sodium, or a small amount of each. Sodium and sugar have an extremely negative correlation together within this dataset.

Since the majority of our dataset is a raw numerical value of a nutritional attribute paired with the corresponding daily percentage value, it is easy to assume that all of our data points are not independent. This is true. However, I ran this model without the daily percentage value columns and each model suffered a 3-10% loss of accuracy. This is why every column is included in the final model.

![](/pairplot.png)
## Methods

### Model Overview
For this problem I wanted to try as many models as I could since sklearn makes it so easy to test models. I had particular hope for the K-Nearest Neighbor Model and the Support Vector Classifier. Since every item within a category of food should be similar to one another, models that plot and split data within an euclidean distance should fare well. I also had particular hope for the Logistic Regression, since it was the first machine learning model I ever used. 

### Strengths and Weaknesses
#### Strengths
Our model has a few strengths. First off, the model has plenty of features to work with. The model has 25 features to predict a single classification. Second, the model itself is very fast and lightweight to run. Since there isn't a lot of data to process, each individual model takes less than a second to compute. Each model is also better at making predictions than a blind guess.
#### Weaknesses
The main weakness to my analysis is the lack of balanced data. We are just shy of 400 data entries. These 400 entries are split across 9 different categories of food. These categories of food are also extremely unbalanced, 1/4 of our entire dataset is just beverages.By changing the random state of X, y split, I am able to make some models 10 points better and others 10 points worse. It's not a good idea to have the accuracy of your models relient on the split that you get from your data. The more balanced and the more data you have, the better and robust a model is. If I were to redo this project, I would gather more data, and focus on balancing the data between the different categories.

### Model Performance
#### Overview
All of the models featured in my analysis are listed below, each with their relative Accuracy, F1, Precision, and Recall Scores. (Random state 713).

* Decision Tree
  * Accuracy Score : 0.6971830985915493
  * F1 Score : 0.7013347116034743
  * Precision Score :  0.7191855380649912
  * Recall Score :  0.6971830985915493

* Random Forest
  * Accuracy Score : 0.7676056338028169
  * F1 Score : 0.7587853406556322
  * Precision Score :  0.7724775553908244
  * Recall Score :  0.7676056338028169

* Gradient Boosting
  * Accuracy Score : 0.7464788732394366
  * F1 Score : 0.7421122806316535
  * Precision Score :  0.7484092498251924
  * Recall Score :  0.7464788732394366

* Logistic Regression
  * Accuracy Score : 0.7816901408450704
  * F1 Score : 0.7696559800064551
  * Precision Score :  0.7791710791062163
  * Recall Score :  0.7816901408450704

* K-Nearest Neighbor
  * Accuracy Score : 0.7816901408450704
  * F1 Score : 0.7821055600079996
  * Precision Score :  0.7932466008170234
  * Recall Score :  0.7816901408450704

* Support Vector Classifier
  * Accuracy Score : 0.8169014084507042
  * F1 Score : 0.8146139005328851
  * Precision Score :  0.8187933213751699
  * Recall Score :  0.8169014084507042
  
Our best models (Logistic Regression, K-Nearest Neighbors, and Support Vector Classifier), all scored around 80%. This was lower than I originally anticipated. When I look at the confusion matrix for our best model (Support Vector Classifier), we can see that our model really struggled with the last column in the matrix. That last column is our "Snacks and Sides'' column. The category "snacks and sides" included the widest variety of food items. The category included salty fries and sweet pretzels. These two food items are vastly different. It is no surprise that our model struggled with classifying this type of food.

(Here are the column names in order -\
0 - Beef and Pork\
1 - Beverages\
2 - Breakfast\
3 - Chicken and Fish\
4 - Coffee and Tea\
5 - Desserts\
6 - Salads\
7 - Smoothies and Shakes\
8 - Sides and Snacks\
)

[[11  0  0  0  0  0  1  0  2]\
 [ 0 16  0  0  2  0  0  0  0]\
 [ 0  0 21  0  0  0  0  0  2]\
 [ 0  0  1 14  0  0  0  0  1]\
 [ 0  6  0  0 35  0  0  1  0]\
 [ 0  1  0  0  1  0  0  0  0]\
 [ 2  0  0  0  0  0  6  0  0]\
 [ 0  0  0  0  3  0  0  9  0]\
 [ 1  0  2  0  0  0  0  0  4]]
 
#### Different Random States
Like what was stated earlier, the lack of data in our model allowed for widely different accuracy scores based on the split of the data. If we used random state 2, both Random Forest and K-Nearest Neighbor have their accuracies well above 80%. Our best model from above, SVC, has it's accuracy dip down to 76%.

* Random Forest
  * Accuracy Score : 0.8450704225352113
  * F1 Score : 0.8401488754999525
  * Precision Score :  0.8641574481973572
  * Recall Score :  0.8450704225352113

* Fitting KNN
  * Accuracy Score : 0.8380281690140845
  * F1 Score : 0.8372895552390167
  * Precision Score :  0.8455404771248752
  * Recall Score :  0.8380281690140845
  
* Fitting SVC
  * Accuracy Score : 0.7676056338028169
  * F1 Score : 0.7567580291021508
  * Precision Score :  0.7631443204682641
  * Recall Score :  0.7676056338028169
 
#### Voting Classifier
To finish everything off, I decided to make a voting classifier model to hopefully combat the lack of balanced data. In this model, I used the best individual models from our random state 2 run (KNN, Logistic Regression, Gradient Boosting, and Random Forest).

This model was able to achieve an accuracy higher than any of their composite parts.

* Voting Classifier
  * Accuracy Score:  0.852112676056338
  * F1 Score:  0.8476444832659763
  * Recall Score:  0.852112676056338
  * Precision Score: 0.8626825052881391
  
[[13  0  0  0  0  0  0  0  0]\
 [ 0 14  0  0  2  0  0  0  0]\
 [ 0  0 18  2  0  0  0  0  0]\
 [ 1  0  2  9  0  0  0  0  1]\
 [ 0  3  0  0 46  0  0  0  0]\
 [ 0  1  0  0  1  1  0  0  0]\
 [ 2  0  0  0  0  0  5  0  0]\
 [ 0  0  0  0  2  0  0  9  0]\
 [ 2  0  0  1  1  0  0  0  6]]

When we look at the confusion matrix, we can see that the last column (snacks and sides), is much better predicted.

# Conclusion
In conclusion, I would consider the SuperSize Me project a success. In this project, I was able to explore the menus and nutritional data of 3 of my most frequented restaurants. From this nutritional data, I was able to build a few models to fairly accurately predict what type of food was being served based on the nutritional data that was contained within. Our models struggled with correctly classifying "Sides and Snacks", but was able to shine when given the right split of data. Our voting classifier was actually able to produce a result greater than its composite parts.

With some more tuning, this model could be great at classifying a large dataset of nutritional data. If a researcher wanted to run an analysis on some uncategorized nutritional data, a model like this could aid them in the classification of their data. (Although a neural network or some unsupervised model might be able to do this better and at a greater accuracy).

## Limitations
To reiterate what was mentioned in the weakness section of this report, the greatest limitation to this project was the data. The model was only fed 400 or so entries. These 400 entries were then split among 9 unbalanced classifications. If more data was cleaned and collected, then our model would be able to predict much more accurately. Not only would the model accuracy be better, but the model would also be more robust, making it less susceptible to differences in the random state of the X y split.

## Tangent
A model similar to this could be used by companies when 3d printed food becomes more widely available. Many of these machines take in raw components to print food. A model that takes raw nutritional data as inputs could pair nicely with a machine that does the same.

# Data Credits
In n Out - https://www.in-n-out.com/pdf/nutrition_2010.pdf
Red Robin - https://www.redrobin.com/pages/nutrition/
McDonalds - https://www.kaggle.com/mcdonalds/nutrition-facts


