# Molecular_Properties_Prediction
Data sourced from Kaggle Competition: [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/data)

**Author**:
- [Hyunwook Paul Shin](https://github.com/hps1795)

<img src="Image/molecules.png" width="100%" height="100%">

## Overview
This notebook contains the steps of solving the molecular properties prediction involving the molecular structural data. The aim of the project is to create a model that can provide the accurate prediction on the target variable, scalar coupling constant. Since this is a prediction on the continous quantity output, the problem is defined as the regression problem. Then, the metrics are set as R-squared score, root mean squared error, and mean absolute error to evaluate the regression model: 

***

## Business Problem
Nuclear Magnetic Resonance (NMR) is a core technique used to understand the structure and interactions of the molecules and proteins. NMR is being utilized by the researchers in pharmaceutical and chemical field worldwide. The NMR performance is largely dependent on the accurate prediction on the variable, scalar coupling constant (SCC).

SCC is a magnetic interaction (also called 'coupling') between two atoms. It is the feature that provides information on the connectivity of chemical structure which is used to explain the interaction between the molecules in NMR. However, the constraint in SCC calculation limits the application of this technique; the calculation length takes from days to even weeks for one molecule when the structural information of the molecule is an input and the cost of calculation is expensive.

Therefore, creating a model that can accurately predict the SCC will allow the NMR to be applicable for research in daily basis. The model will allow the phamceutical researchers to gain insight on how the molecular structure affects the properties and behavior faster and cheaper, and accelerate the innovation in inventing and designing new drugs.

***

## Data Understanding
**Data sourced from Kaggle Competition: [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/data)**

Two dataset we will be using are:

- train.csv
    - the training set, where the first column (molecule_name) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (atom_index_0) and third column (atom_index_1) is the atom indices of the atom-pair creating the coupling and the fourth column (scalar_coupling_constant) is the scalar coupling constant that we want to be able to predict

- structures.csv 
    - this file contains the same information as the individual xyz structure files, but in a single file

Distribution of the target variable is displayed below:
<img src="Image/Target_Distribution.png">
The distribution is right skewed, which means the target variable needs to be normalized.

After Cubic Rooted Transformation, target variable is more normalized as shown below.
<img src="Image/Normalized_Target_Distribution.png">

***

## Data Preparation 
For data preparation, following tasks were done:
- Merge train_df and structures_df
- Create new features, such as distance and angle of the bond
- Normalize the target variable
- Check the multicollinearity of each columns
- Split the dataset into train, test, and validation set
- Perform preprocessing

For Normalization, cubic root transformation was used. Logistic transformation and square root transformation is also the good option in handling the skewed distribution. However, since the logarithm and square root is only defined for positive numbers, you can't take the logarithm of negative values. This means logistic transformation cannot be applied on our target variable which contains negative values.

Rule of thumb for multicolinearity is that severe multicollinearity may be present if the correlation > 0.8. None of the features had coeficient higher than 0.8, and, therefore, it was concluded that dataset does not have severe multicolinearity. Multicolinearity was found by using the heatmap shown below. 

<img src="Image/Multicollinearity.png">


For the preprocessing, the columns are divided into two different columns: numerical and categorical. Numerical columns are scaled with MinMaxScaler, while categorical columns are encoded using LabelEncoder.

***

## Modeling Result

Our final model is a Keras Sequential Model. The report of the result on holdout test set is displayed below.


<img src="Image/Final_Model_Results.png">

- R2 Score: 0.9461544983983261
- RMSE: 0.45984758219441946
- MAE: 0.26220035989513957
- Prediction time: 89.96339583396912s for 931816 predictions

***

## Conclusions

Based on the results on the test set:

- The R2 score on the test set is 0.946. This means the predicted target variable on the test set shows high positive correlation with the actual test target variable. Also, R2 score of 0.946 means approximately 94.6% of the observed variation can be explained by the final keras model.
- RMSE of 0.460 and MAE of 0.262 are both low values. This means the final model can relatively predict the target variable accurately.
- Metrics for the test set is very close those of the train and validation set. This means the final model is generalized and available to be applied to the real-world coupling constant prediction without biased.
- When neccessary informations (features in train_df2) are provided, final model can give accurate prediction on the target variable in less than a second for each combination of the atom inside the molecule: It took about 90 seconds for more than 90k predictions on test set. This will significantly reduce the cost and time it takes for predicting the scalar coupling constant, which takes days to even weeks for one molecule when traditional calculation method is used. Therefore, I strongly recommend researchers to apply this model to their NMR research process to speed up their research process.


***

## Next Steps

- **Data Collection and Feature Engineering:** Additional feature engineering can be done to enhance the accuracy of the model prediction even better. Features like type of hybridization, which is the new hybrid created by mixing different atoms, affects the structure and the properties of the molecule, which is definitely a useful factor for explaining the property of the molecule. Also, gathering even more data can even create even more generalizable and accurate model. Therefore, more data scrapping on different molecules is recommended.


- **Divide into Subset**: The analysis and prediction was made on the whole train_df in this project: overall performance does not explain how model works on each type of couplings. Therefore, dividing up the data into subset based on 8 different coupling types will allow to fine-tune model specifically targeting one coupling type. Also, models we previously used might perform better on one type while shows worse performance on other types. By dividing the dataset and model them, we can build further insight on which model and hyperparameters works best for each type of couplings and enhance the prediction accuracy even further.


***

## Information
Information of the modules used and their versions to recreate this project:

- jupyter: 1.0.0
- Python: version 3.8.5
- numpy: 1.21.4
- pandas: 1.1.3
- seaborn: 0.11.0
- matplotlib: 3.3.1
- scikit-learn: 0.23.2 
- xgboost: 1.2.1
- lightgbm: 3.3.1
- catboost: 1.0.3
- tensorflow: 2.3.1
- keras: 2.4.3

Dataset for the project is over 100MB so it cannot be uploaded to the GitHub. For the dataset, proceed to the following link: https://www.kaggle.com/c/champs-scalar-coupling/data. One who wants to download the dataset needs to sign up for the competition. After agreeing on the acknowledgement, one will be able to download the dataset files. You can choose to download the files you desire to use only or simply just download all. When you click download all, you will be dowloading the zip file, **champs-scalar-coupling.zip**. Unzip the files and move data into the Data/ directory of this repository.
***

## Repository Structure

```
├── Data                                <- Empty Data Folder (Read Information section for the guide to download the dataset) 
├── Images                              <- Folder containing graphs and images from notebooks and presentation
│   └── ...
├── .gitignore                          <- Data file was large and ignored
├── Final Notebook.ipynb                <- Narrative documentation of project in Jupyter notebook
├── README.md                           <- Abstract Summary of the Final Notebook 
└── presentation.pdf                    <- PDF version of project presentation
``` 
