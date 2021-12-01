# Molecular_Properties_Prediction
Data sourced from Kaggle Competition[Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/data)

**Author**:
- [Hyunwook Paul Shin](https://github.com/hps1795)
- 
<img src="Image/molecules.png" width="80%" height="80%">

## Overview
This notebook contains the steps of solving the molecular properties prediction involving the molecular structural data. The aim of the project is to create a model that can provide the accurate prediction on the target variable, scalar coupling constant. Since this is a prediction on the continous quantity output, the problem is defined as the regression problem. Then, the metrics are set as R-squared score, root mean squared error, and mean absolute error to evaluate the regression model: 

***

## Business Problem
Nuclear Magnetic Resonance (NMR) is a core technique used to understand the structure and interactions of the molecules and proteins. NMR is being utilized by the researchers in pharmaceutical and chemical field worldwide. The NMR performance is largely dependent on the accurate prediction on the variable, scalar coupling constant (SCC).

SCC is a magnetic interaction (also called 'coupling') between two atoms. It is the feature that provides information on the connectivity of chemical structure which is used to explain the interaction between the molecules in NMR. However, the constraint in SCC calculation limits the application of this technique; the calculation length takes from days to even weeks for one molecule when the structural information of the molecule is an input and the cost of calculation is expensive.

Therefore, creating a model that can accurately predict the SCC will allow the NMR to be applicable for research in daily basis. The model will allow the phamceutical researchers to gain insight on how the molecular structure affects the properties and behavior faster and cheaper, and accelerate the innovation in inventing and designing new drugs.

***

## Data Understanding
**Data sourced from Kaggle Competition[Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/data)**

Two dataset we will be using are:

- train.csv
    - the training set, where the first column (molecule_name) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (atom_index_0) and third column (atom_index_1) is the atom indices of the atom-pair creating the coupling and the fourth column (scalar_coupling_constant) is the scalar coupling constant that we want to be able to predict

- structures.csv 
    - this file contains the same information as the individual xyz structure files, but in a single file

Distribution of the Target Variable is displayed below:

The distribution is right skewed, which means the target variable needs to be normalized.

***

## Data Preparation 
For data preparation following tasks were done:
- Merge train_df and structures_df
- Create new features, such as distance and angle of the bond
- Normalize the target variable
- Check the multicollinearity of each columns
- Split the dataset into train, test, and validation set
- Perform preprocessing

For Normalization, cubic root transformation was used. Logistic transformation and square root transformation is also the good option in handling the skewed distribution. However, since the logarithm and square root is only defined for positive numbers, you can't take the logarithm of negative values. This means logistic transformation cannot be applied on our target variable which contains negative values.

Multicolinearity was found by using the heatmap shown below. Rule of thumb for multicolinearity is that severe multicollinearity may be present if the correlation > 0.8.
None of the features had coeficient higher than 0.8, and, therefore, it was concluded that dataset does not have severe multicolinearity.

For the preprocessing, the columns are divided into two different columns: numerical and categorical. Numerical columns are scaled with MinMaxScaler, while categorical columns are encoded using LabelEncoder.

***

## Modeling Result

Our final model is a Keras Sequential Model. The report of the result is displayed below.


***

## Conclusions




***

## Information


## Repository Structure

```
├── Images                              <- Folder containing graphs and images from notebooks and presentation
│   └── ...
├── Notebooks                           <- Directory containing individual group members' notebooks        
│   └── ...
├── Final Notebook.ipynb                <- Narrative documentation of project in Jupyter notebook
├── README.md                           <- Top-level README 
└── presentation.pdf                    <- PDF version of project presentation
``` 
