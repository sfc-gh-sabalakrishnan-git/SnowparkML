# Loan Default Prediction Using Snowpark ML & Model Registry

In this post, we explore an advanced implementation of loan default prediction by leveraging Snowpark ML and the efficiency of the XGBoost algorithm within the Snowflake environment. This approach builds upon our previous exploration of deep learning techniques for similar predictions, transitioning from TensorFlow and Keras (https://medium.com/p/78a15b196e65) to a more scalable and robust method suited for handling vast datasets in the data cloud.

## Initializing the Snowpark Session

First, we establish a Snowpark session, connecting to Snowflake to utilize its compute clusters and data storage capabilities. This session is crucial for all subsequent operations, including data manipulation, model training, and interaction with the Snowpark Model Registry.

connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

## Data Preparation
The robustness of Snowpark allows for efficient data handling and computational operations directly within Snowflake's environment. The dataset we are going to use is from Kaggle. To simplify, we keep the loan data downloaded from Kaggle (https://www.kaggle.com/datasets?search=loan+dataset)  in Snowflake's internal staging. You can also use external staging with S3 or store the data in a Snowflake table.

session.sql("LS @Loan_Data;").show()

 
```
#Create a Snowpark DataFrame that is configured to load data from the CSV file
#We can now infer schema from CSV files.
loandata_df = session.read.options({"field_delimiter": ",",
                                    "field_optionally_enclosed_by": '"',
                                    "infer_schema": True,
                                    "parse_header": True}).csv("@Loan_Data")
loandata_df.columns
```
In this dataset, except for the “PURPOSE” column, all are numeric, which simplifies our task. Moreover, all numeric columns are in a range that does not require further manipulation, except converting them to Double type for Snowflake supported types.

```
#Loop through a list of column names and cast each column to DoubleType in the 'loandata_df' DataFrame.
for colname in ["CREDIT_POLICY", "INT_RATE", "INSTALLMENT", "LOG_ANNUAL_INC", "DTI", "FICO","DAYS_WITH_CR_LINE","REVOL_BAL","REVOL_UTIL","INQ_LAST_6MTHS","DELINQ_2YRS","PUB_REC","NOT_FULLY_PAID"]:
    loandata_df = loandata_df.with_column(colname, loandata_df[colname].cast(DoubleType()))
list(loandata_df.schema)
```
```
[StructField('PURPOSE', StringType(16777216), nullable=True),
 StructField('CREDIT_POLICY', DoubleType(), nullable=True),
 StructField('INT_RATE', DoubleType(), nullable=True),
 StructField('INSTALLMENT', DoubleType(), nullable=True),
 StructField('LOG_ANNUAL_INC', DoubleType(), nullable=True),
 StructField('DTI', DoubleType(), nullable=True),
 StructField('FICO', DoubleType(), nullable=True),
 StructField('DAYS_WITH_CR_LINE', DoubleType(), nullable=True),
 StructField('REVOL_BAL', DoubleType(), nullable=True),
 StructField('REVOL_UTIL', DoubleType(), nullable=True),
 StructField('INQ_LAST_6MTHS', DoubleType(), nullable=True),
 StructField('DELINQ_2YRS', DoubleType(), nullable=True),
 StructField('PUB_REC', DoubleType(), nullable=True),
 StructField('NOT_FULLY_PAID', DoubleType(), nullable=True)]

```
### Column Descriptions

1.	credit_policy: Indicates whether the borrower meets the credit underwriting criteria of LendingClub. This might be a binary indicator (e.g., 1 for meeting criteria, 0 otherwise).
2.	purpose: Describes the purpose of the loan as reported by the borrower. Common purposes include debt consolidation, credit card refinancing, home improvement, major purchase, small business investment, educational expenses, etc.
3.	int_rate: The interest rate of the loan as a percentage. This reflects the cost of borrowing to the loanee.
4.	installment: The monthly payment owed by the borrower if the loan originates. This is calculated based on the loan amount, term, and interest rate.
5.	log_annual_inc: The natural logarithm of the self-reported annual income of the borrower. Using the logarithm helps in normalizing the distribution of income values.
6.	dti: The debt-to-income (DTI) ratio of the borrower at the time of application. It is a measure of the borrower's monthly debt payments divided by their gross monthly income, expressed as a percentage.
7.	fico: The FICO credit score of the borrower. This is a critical factor used to evaluate the credit risk of the borrower.
8.	days_with_cr_line: The number of days the borrower has had a credit line. This might reflect the borrower's experience and history with credit.
9.	revol_bal: The borrower's revolving balance, which indicates the amount of credit the borrower is using relative to their available revolving credit.
10.	revol_util: The borrower's revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit, expressed as a percentage.
11.	inq_last_6mths: The number of inquiries into the borrower's credit during the past six months. Multiple inquiries can indicate that a borrower is seeking several loans or lines of credit, which may impact their credit score.
12.	delinq_2yrs: The number of instances the borrower was delinquent on any credit account for 30+ days within the past two years.
13.	pub_rec: The number of derogatory public records the borrower has, such as bankruptcy filings, tax liens, or judgments.
14.	not_fully_paid: This column indicates whether the loan was not fully repaid. It's likely a binary indicator, where 1 might mean the loan is not fully paid off, and 0 means it is.


## Ordinal Encoding

Ordinal encoding is applied to the "PURPOSE" column for several reasons, with the choice of this particular column being strategic based on the data's nature and the requirements of subsequent analysis or machine learning models.

Why Ordinal Encoding?
1.	Numerical Representation: Machine learning models inherently require numerical input. Ordinal encoding converts the categorical "PURPOSE" column into a numerical format that can be directly used by these models.
2.	Preserving Order (When Applicable): Although the "PURPOSE" categories might not have a natural order that is universally agreed upon, in some analytical contexts, there could be an implied or practical order. For example, one might argue that "EDUCATIONAL" loans have different risk profiles compared to "SMALL_BUSINESS" loans. Ordinal encoding allows for the possibility of assigning an order if it is deemed analytically meaningful, even though, in this specific instance, the order is based on the array's sequence rather than an inherent property of the data.
3.	Efficiency: Ordinal encoding is more space-efficient compared to one-hot encoding, as it creates a single new column rather than multiple columns for each category. This can be particularly beneficial when dealing with datasets with a large number of categories or when operating under storage or computational constraints.

```
from snowflake.snowpark.functions import col
unique_PURPOSE=loandata_df.select(col("PURPOSE")).distinct()
unique_PURPOSE.show()
```
```
----------------------
|PURPOSE           |
----------------------
|DEBT_CONSOLIDATION  |
|CREDIT_CARD         |
|ALL_OTHER           |
|HOME_IMPROVEMENT    |
|SMALL_BUSINESS      |
|MAJOR_PURCHASE      |
|EDUCATIONAL         |
```

### Why Choose the "PURPOSE" Column?
1.	Categorical with Multiple Levels: The "PURPOSE" column is a prime candidate for encoding because it contains categorical data with multiple distinct levels. Encoding transforms these textual descriptors into a machine-readable format.
2.	Relevance to Analysis/Modeling Goals: The purpose of a loan is likely to have a significant impact on the outcome of interest, such as loan default risk. By encoding this column, we ensure that this potentially predictive information is included in the model in a usable form.
3.	Varied Categories: The "PURPOSE" column contains a variety of reasons for which the loans were taken, such as debt consolidation, credit card payments, home improvement, etc. Each of these reasons can influence the loan's risk profile differently. Encoding these categories allows the model to potentially discern patterns related to each loan purpose.


categories = {"PURPOSE": np.array(["DEBT_CONSOLIDATION", "CREDIT_CARD", "ALL_OTHER", "HOME_IMPROVEMENT", "SMALL_BUSINESS","MAJOR_PURCHASE","EDUCATIONAL"]) }

#Initialize an OrdinalEncoder to encode the 'PURPOSE' column according to the specified categories.
snowml_oe = snowml.OrdinalEncoder(input_cols=["PURPOSE"], output_cols=["PURPOSE_OE"], categories=categories)

#Fit the OrdinalEncoder to the loandata_df and transform it, encoding the 'PURPOSE' column.
ord_encoded_loandata_df = snowml_oe.fit(loandata_df).transform(loandata_df)


#Drop the original 'PURPOSE' column from the DataFrame after encoding.
#Rename the encoded column 'PURPOSE_OE' back to 'PURPOSE' for clarity in the final DataFrame.
ord_encoded_loandata_df=ord_encoded_loandata_df.drop("PURPOSE")
df_clean = ord_encoded_loandata_df.rename(col("PURPOSE_OE"), "PURPOSE")

In summary, ordinal encoding for the "PURPOSE" column is a strategic choice to transform categorical data into a numerical format that is suitable for machine learning models, making the data analysis-ready while potentially capturing the predictive power of the loan purpose in relation to the outcome of interest.


## Handling Class Imbalance by Over-sampling

Class imbalance can lead to models that are overly biased towards the majority class, under-performing in accurately predicting the minority class instances due to the lack of sufficient data to learn from. This can be particularly problematic in applications like loan default prediction, where failing to identify potential defaults (the minority class) could have significant financial implications.

count_df = df_clean.group_by("NOT_FULLY_PAID").agg(count("*").alias("count"))
total_count = df_clean.count()
proportion_df = count_df.with_column("proportion", count_df["count"] / lit(total_count))
proportion_df.show()

NOT_FULLY_PAID is  the Label column


Other Methods to Handle Class Imbalance:
1.	Under-Sampling: Reducing the number of instances in the majority class to match the minority class count. While it can balance the dataset, it may also lead to the loss of potentially valuable data.
2.	Synthetic Data Generation: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) generate synthetic instances of the minority class by interpolating existing minority instances, enhancing model training without repeating exact copies.
3.	Cost-sensitive Learning: Adjusting the model's learning algorithm to penalize misclassifications of the minority class more than those of the majority class to force the model to pay more attention to the minority class.
4.	Ensemble Methods: Using ensemble learning techniques, such as boosting or bagging, with a focus on balancing class distribution within each ensemble's subset of data.


## Hyperparameter tuning and Building the Model

Hyperparameter optimization is crucial to enhance the performance of models. The process involves tuning parameters like max_depth, learning_rate, and n_estimators to find the optimal settings for our model. The default parameters of a model might not be ideal for all types of data or problems. By tuning hyperparameters, we can significantly improve a model's accuracy, efficiency, and generalization ability to unseen data, ensuring the model is well-suited for its specific application.

```

model = XGBClassifier()
train_df, test_df = df_final.random_split(weights=[0.9, 0.1], seed=0)

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300]
}
```
Parameters explanation: 

max_depth: Determines the maximum depth of the trees. Deeper trees can model more complex patterns but might lead to overfitting. Values explored are 3, 4, and 5, aiming to find a balance between model complexity and generalization.

learning_rate: Controls the step size at each iteration while moving toward a minimum of a loss function. A lower rate requires more iterations but can achieve a more accurate model. Tested rates are 0.01, 0.1, and 0.2, to evaluate the trade-off between convergence speed and model performance.

n_estimators: Specifies the number of trees in the ensemble. More trees can lead to better performance but also increase computation time. The grid search tests 100, 200, and 300 trees to find an optimal number for the ensemble size.
```

grid_search=GridSearchCV(
    estimator=XGBClassifier(),
    param_grid=param_grid,
    n_jobs=-1,
    scoring="accuracy",
    input_cols=train_df.drop("NOT_FULLY_PAID").columns,
    label_cols="NOT_FULLY_PAID",
    output_cols="PREDICTION",
)
```
### Grid Search:
Grid search is a brute-force method for hyperparameter optimization. It systematically creates and evaluates models for each combination of the parameter grid specified. This approach ensures that every possible combination is tested but can be computationally expensive, especially with large datasets and many parameters.

grid_search.fit(train_df)

Other Types of Hyperparameter Optimizations:

Random Search: Samples parameter combinations randomly. This method is more efficient than grid search when dealing with a large number of hyperparameters.

Bayesian Optimization: Uses a probabilistic model to predict the performance of parameter combinations and selects new combinations to test based on past results, optimizing both exploration of the parameter space and exploitation of known good areas.

Gradient-based Optimization: Applies gradient descent or similar methods to optimize hyperparameters, particularly useful when the parameters are continuous.

Evolutionary Algorithms: Mimic the process of natural selection to iteratively select, mutate, and combine parameters to find optimal solutions over generations.

## Evaluating Model Performance: Accuracy and Confusion Matrix Analysis

We evaluate the model's performance through accuracy and confusion matrix analysis, ensuring the model is effective in making accurate predictions. Use the best model found by grid search to make predictions on the test dataset and store the results in `predictions_df`.
```
predictions_df = grid_search.predict(test_df)

predictions_df.select("NOT_FULLY_PAID", "PREDICTION").show()

Lets check the prediction accuracy 
traning_accuracy_score=accuracy_score(
    df=predictions_df,
    y_true_col_names=LABEL_COLS,
    y_pred_col_names=OUTPUT_COLS
)
traning_accuracy_score

0.901609
```
Prediction accuracy score of 0.907797 indicates that the model accurately predicted the correct outcome for approximately 90.78% of the cases in the test dataset. This high accuracy score suggests that the model is highly effective in making predictions for this particular task.


```
traning_confusion_matrix=confusion_matrix(
    df=predictions_df,
    y_true_col_name=LABEL_COLS[0],
    y_pred_col_name=OUTPUT_COLS[0]
)
traning_confusion_matrix
```
The confusion matrix output represents the performance of the classification model:
```
Top-left (694): True Negatives (TN) - The number of instances correctly predicted as negative (not fully paid).
Top-right (107): False Positives (FP) - The number of instances incorrectly predicted as positive (fully paid) when they are actually negative.
Bottom-left (42): False Negatives (FN) - The number of instances incorrectly predicted as negative when they are actually positive.
Bottom-right (773): True Positives (TP) - The number of instances correctly predicted as positive.
This matrix helps in understanding the model's ability to correctly or incorrectly classify instances into 'fully paid' or 'not fully paid' categories
```
Model Management and Deployment with Model Registry

with Snowpark ML's model registry, we have a Snowflake native model versioning and deployment framework. This allows us to log models, tag parameters and metrics, track metadata, create versions, and ultimately execute batch inference tasks in a Snowflake warehouse or deploy to a Snowpark Container Service.

Lets Log the model to the Snowflake ML Registry with specified details, including model name, version, requirements, and comments.
Additionally, include model performance metrics (accuracy score and confusion matrix) and a sample of input data for reference or later use.

```
reg=Registry(session=session,  database_name=session.get_current_database(),schema_name=session.get_current_schema())

mv=reg.log_model(
    grid_search,
    model_name="xgb_loan_default_prediction",
    version_name="v3",
    pip_requirements=["packaging"],
    comment="simple  XGB model to predict loan default",
    metrics={
    "training_accuracy_score":traning_accuracy_score,
    "training_confution_matrix":traning_confusion_matrix.tolist()
    },
    sample_input_data=train_df
)
```
Use the logged model for predicting loan default, demonstrating the practical application of registered models in making accurate predictions on new data.
```
rf_mv1=model_list[2].version('V3')
pred_df1=rf_mv1.run(test_df,function_name='predict')
pred_df1['NOT_FULLY_PAID','PREDICTION'].show()
```
## Conclusion
This process effectively showcases the use of Snowflake's Snowpark ML model registry for managing and deploying machine learning models. By logging the model with detailed metadata and performance metrics, we ensure transparency and ease of model management. Deploying a specific version of the model for prediction exemplifies the registry's capability to streamline model lifecycle management, facilitating a robust and scalable framework for operationalizing machine learning models.




