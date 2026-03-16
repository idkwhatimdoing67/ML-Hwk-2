#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from scipy.stats import shapiro
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, sep=';') #Adjusted for csv type
        self.processed_data = None
    
    # Idea to check p-value to determine pre-processing route.
    # New to shapiro method so this will be a test
    def preprocess(self):
        df = self.raw_input.copy() # Copy data to preprocessing df
        
        # Iterate through df, working on a col at a time
        for col_name in df.columns:
            col_data = df[col_name]

            # Handle numeric columns first:
            
            # Change numpy dtype check to pandas
            if pd.api.types.is_numeric_dtype(col_data):
                # First use IQR to test if col is worth p-value test
                Q1 = col_data.quantile(.25)
                Q3 = col_data.quantile(.75)
                IQR = Q3 - Q1
                outlier_mask = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR)))
                
                # Decision logic
                if outlier_mask.sum() / len(col_data) > 0.05:
                    # If we have too many outliers, then no use in p value test
                    scaler = RobustScaler()
                else:
                    # Now we check for normality to see if we use normal or min/max scaling
                    stat, p = shapiro(col_data.dropna().iloc[:5000])
                    if p > 0.05:
                        scaler = StandardScaler() # Proper normalization
                    else:
                        scaler = MinMaxScaler() # Still skewed
                
                # Once chosen, we transform for the scaler
                df[col_name] = scaler.fit_transform(df[[col_name]])
                
            # Handle categorical data
            else:
                # Binary means we can use labelencoder
                if col_data.nunique() <= 2:
                    le = LabelEncoder()
                    df[col_name] = le.fit_transform(col_data)
                else:
                    # More than 2, we use One-Hot Encoding
                    dummies = pd.get_dummies(df[col_name], prefix=col_name)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col_name, axis=1, inplace=True)
        self.processed_data = df
        self.processed_data = df.astype(float) # Should turn T/F into 1/0
        return self.processed_data

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        
        # Provided data prep
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y) 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        
        # Storage for results
        results = []
        plt.figure(figsize=(12,8))
        
        # Iterate through all combinations of hypers
        for act in activations:
            for lr in learning_rate:
                for layers in num_hidden_layers:
                    for epochs in max_iterations:
                        
                        # Setup our hidden layers
                        # Note: this test uses 10 as arbitrary number, subject to change
                        hidden_config = tuple([10] * layers)
                        
                        # Actual MLPClassifier model setup, we intialize our loop vars so we can iterate thru all
                        model = MLPClassifier(
                            hidden_layer_sizes = hidden_config,
                            activation = act,
                            learning_rate= lr,
                            max_iter = epochs,
                            random_state = 42 
                        ) 
                        
                        # Train
                        model.fit(X_train, y_train)
                        
                        # Capture training stats
                        train_acc = model.score(X_train, y_train)
                        train_error = model.loss_
                        
                        # Store data for output table, append each iteration/result
                        hyperparams = f"{act}, lr={lr}, layers={layers}, epochs={epochs}"
                        results.append({
                            "Hyperparameters": hyperparams,
                            "Train Acc": f"{train_acc:.2%}",
                            "Train Error": f"{train_error:.4f}"
                        })
                        
                        # Store history for plot
                        plt.plot(model.loss_curve_, label=f"{act} | lr {lr} | L{layers}")
        
        # Display training table
        results_df = pd.DataFrame(results)
        print("\nTraining Phase Results")
        print(results_df.to_string(index=False))
        
        # Plot setup
        plt.title("Model History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (Error)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show()

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("student-mat.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()

