# EPFL Machine Learning Course CS-433 Project 1

This is the solution for the first project for the Machine Learning course at EPFL.

The  purpose  of this  project  was  to  showcase  how  machine  learning  can  help  with the task of finding the Higgs boson using an effective classification methods for a vector of features that represent the decay signature of a collision effect.

Additionally, this project does exploratory data analysis to understand the dataset and the features, feature processing and engineering to clean the dataset and extract more meaningful information, use machine learning methods on real data, analyze the model and generate predictions using those methods.

## Best Submission

We reached 80.3% accuracy on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/) with multiple submissions. We choose submission ID **22324** as our final one. The script `run.py` generates the same predictions.

## Project Structure

The project is structured as follows:

    .
    ├── data                    # Train and test datasets
    ├── cross_validation.py     # Hyperparameter tuning
    ├── helpers.py              # Helper functions for loading data, making predictions, and creating submission files
    ├── implementations.py      # Implementations of all the ML models
    ├── preprocessing.py        # Functions for preprocessing
    ├── run.py                  # Main script
    ├── tune_params.py          # Script used to find the optimal hyperparameters with grid search
    └── README.md               # README

## Dependencies

The only required dependencies are:

```
NumPy 1.17.2
```

## Finding Hyperparameters

You can run `tune_params.py` to see how we chose the optimal hyperparameters. If you are interested in finding optimal hyperparameters, you can see the coefficients in `run.py`. If you want to know more about how we found them, take a look at `cross_validation.py`. It contains the implementation for grid search over the hyperparameter space and later uses cross-validation to obtain the accuracy.

## Running

To build the models, the train and test datasets are required. Both of them are compressed to save space. Before executing the main script, unzip the data. After successfully unzipping the data, run the project by simply executing `python3 main.py`. This will output the predictions in a CSV file inside the `data` folder.

## Authors

* [Ljupche Milosheski](ljupche.milosheski@epfl.ch)
* [Manjot Singh](manjot.singh@epfl.ch)
* [Mladen Korunoski](mladen.korunoski@epfl.ch)
