## Python Package for challenge ##


## Package configuration
```
├── data-science-test                        <- All data and documentations regarding the project
├── data_science_test_AT                     <- Ingestion, Preparation, Modeling, Exposition folder with all necessary methods
│   ├── loader.py                               <- methods used to load the data 
│   ├── preprocess.py                           <- methods used to preprocess the data 
│   ├── train.py                                <- methods used to train the model
│   ├── predict.py                              <- methods used to call the model
│   ├── utils_.py                               <- methods used to save 'static' methods
│   ├── wrappers.py                             <- methods wrappers in case of shell files use (to be updated)
│   ├── prepare.py                              <- Contains methods for specific preparation process (to be added)
│   └── api.py                                  <- Contains methods to call api (to be added)
├── output                                  <- Folder contains all results
├── setup.py                                <- Setup file to build python package '''python -m pip install -e .'''
├── main.py                                 <- Main file to 
├── config_paths.yaml                       <- Configuration file used to setup each step : preprocessintg data, training model, calling model, 
├── README.md                               <- README
└── LICENSE
```


## Project goal

In this test, the `requests` datasets contains information about the requests made by group of individuals (or family) to the french emergency housing public service. A sample of the `requests` dataset corresponds to a unique request. The `individuals` datasets contains information about each individual for all requests.
You can use the column `request_id` to link the two datasets.

The goal is to predict the categorical variable `granted_number_of_nights` which represents the number of nights of emergency housing granted to a group. You can train your model on the `requests_train`, the predictions should be made for requests listed in the `requests_test` dataset. The competition score should also be computed on the `requests_test`.

The evaluation metric is given by the `competition_scorer` defined above. It corresponds to a weighted log-loss with weights 1, 10, 100, or 1000 if the `granted_number_of_nights` takes the value 0, 1, 2, or 3 respectively. Thus beware that you will be penalized harder for classification mistakes made on the higher labels.

The score for a random prediction is shown at the end of this notebook (~1.6), your trained models should at least reach a **score below 1** to be significantly better.

## Project Results 

score = 0.57
The success lies in attributing weight during the learning process respectively 1, 10, 100, 1000 to each class. In fact it allocates much weight to small classes and little weight to major class.





