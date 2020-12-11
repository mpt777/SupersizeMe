# SupersizeMe

## Background
The original idea for the project came to me during my experiences backpacking. I've eaten at plenty of Ma and Pa resturants after a long week of hiking. Usually, These resturants are not required to post their nutritional data for the food that they serve, due to their small business size. I originally planned on building a model that would take catagorical attributes about the food (such as they type of food, serving size, and comparable resturants), and return the predicted nutritional data. This approach wouldn't be very accurate given the limited number of features, and the many targets that I hoped to predict. In the end, I decided that simply looking up a similar food item for it's nutritional data online would've been the simplier and more logical approach. The project then pivoted. In this new model, I would take all of the nutritional data about a food item, and precict what type/category a food item belonged to.

The purpose of this model is explore the similarities and differences between the foods we get at resturants. If we supply a machine learning model with all of the nutritional facts of a piece of food, as well as the resturant that the item came from, can it correctly predict what category of food that we purchased (ie, Hamburger, shake, fries, etc).

## Data
### Collection and Cleaning
Since the inspiration of the project came from my backpacking adventures, and my love for a good burger, I chose 3 resturants that encapsulated most types of burgers that I've had before:
  * McDonalds
  * In N Out
  * Red Robin
  
Since large chains of resturants are required to post their nutritional information in a standardized way, it meant the data collection was fairly easy. The data from McDonalds came from a kaggle dataset that I simply downloaded. The data for In N Out and Red Robin both came off PDF's from their respective websites. I've never scraped data off of a PDF before, and I thought it would be a fun challenege to overcome. It turned out to be much more difficult that I first expected.

To scrape the data off of the PDF's, I used the function read_pdf from the package tabula. Tabula's read_pdf creates a list of dataframes. These dataframes usually dont match dimensions, are missing data, and include overlapping data from otehr dataframes. This meant the cleaning process was rather tedious and difficult, especially for the Red Robin dataset.

Of the largest issues I found was Tabula's tendency to combine rows of data into the same column. To combat this, I would shift data over in a dataframe to make space for the column that was combined. I woudl then need to split the contents of the column into 2, and then convert the columns into their correct data type.

### Creation
Our Red Robin data did not include daily percentage points for each of the different nutritional attributes. I used a KNN imputer to find these missing values. I chose this imputation method since the KNN model searches for similar rows in the dataset to choose their values. Since every resturant serves similar items of food, the KNN produced accurate results.

In addition to the imputation of some data, I added a new categorical variable that labeled which resturant each of the items of food originated from.

The data scraped from the PDF's did not include the category of food each item belonged to. I built a function that would search for certain strings in the name of the item to classify which category it belongs to. 

### Exploritory Data Analysis
Since the whole project is exploritory in nature, the actually EDA in the project is small. I used a pairplot to see how each of the features correlated with every other feature. The most interesting thing that can be gleamed from the graphs is the positive correlation that nearly every feature has with every other feature. Some of the correltaions are weak, but most are are still positive.

This makes sense. As a food item gets larger it usually has more of everything, every nutrional feature.

The only place where this relationship isn't true is the relationship between sugar and sodium. For both of these features, there are "two groupings" of data. The first group of data has an extremely positive slope, shooting off rapidly from the other points of data. The second group of data has a much less positive slope. A more interpretable way to approach this relationship, is to say that items have alot of sugar, a lot of sodium, or a small amount of each. Sodium and sugar have an extremely negative correlation together within this dataset.

Since the majority of our dataset is a raw numerical value of a nutrional attribute paired with the corresponding daily percentage value, it is easy to assume that all of our datapoints are not independent. This is true. However, I ran this model without the daily percentage value columns and each model suffered a 3-10% loss of accuracy. This is why every column is included in the final model.

![](/pairplot.png)
## Methods

### Model Overview
For this problem I wanted to try as many models as I could since sklearn makes it so easy to test models. I had particular hope for K-Nearest Neighbor Model and the Support Vector Classifier. Since every item within a category of food shoud be similar to one another, models that plot and split data within an euclidian distance should fair well. I also had particular hope for the Logistic Regression, since it was the first machine learning model I ever used. 

All of the models featured in my analysis are listed below, each with their relative Accuracy, F1, Precision, and Recall Scores.

* Decision Tree
  * Accuracy Score : 0.6971830985915493
 - F1 Score : 0.7013347116034743
 - Precision Score :  0.7191855380649912
 - Recall Score :  0.6971830985915493

 Random Forest
  -Accuracy Score : 0.7676056338028169
  -F1 Score : 0.7587853406556322
  -Precision Score :  0.7724775553908244
  -Recall Score :  0.7676056338028169

Gradient Boosting
Accuracy Score : 0.7464788732394366
F1 Score : 0.7421122806316535
Precision Score :  0.7484092498251924
Recall Score :  0.7464788732394366

Fitting Logistic Regression
Accuracy Score : 0.7816901408450704
F1 Score : 0.7696559800064551
Precision Score :  0.7791710791062163
Recall Score :  0.7816901408450704

Fitting KNN
Accuracy Score : 0.7816901408450704
F1 Score : 0.7821055600079996
Precision Score :  0.7932466008170234
Recall Score :  0.7816901408450704

Fitting SVC
Accuracy Score : 0.8169014084507042
F1 Score : 0.8146139005328851
Precision Score :  0.8187933213751699
Recall Score :  0.8169014084507042

Our best models (Logistic Regression, K-Nearest Neighbors, and Support Vector Classifier), all scored around 80%. This was lower than I originally anticipated

### Strengths and Weaknesses

### Model Performance

# Conclusion
