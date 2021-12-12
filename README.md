## Python Package for challenge ##
This package has been designed in a course of 3 days and opens the door to many improvements. 
Feel free to contact me if you see any.

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
While the structure could be reused for other competition and projects, this is not aimed to be a generic project. This is the main reason of not using oriented object programming.
However I might upload in the SKELETON project a reusable framework for data science project and deployement.

## Project goal

In this test, the `requests` datasets contains information about the requests made by group of individuals (or family) to the french emergency housing public service. A sample of the `requests` dataset corresponds to a unique request. The `individuals` datasets contains information about each individual for all requests.
You can use the column `request_id` to link the two datasets.

The goal is to predict the categorical variable `granted_number_of_nights` which represents the number of nights of emergency housing granted to a group. You can train your model on the `requests_train`, the predictions should be made for requests listed in the `requests_test` dataset. The competition score should also be computed on the `requests_test`.

The evaluation metric is given by the `competition_scorer` defined above. It corresponds to a weighted log-loss with weights 1, 10, 100, or 1000 if the `granted_number_of_nights` takes the value 0, 1, 2, or 3 respectively. Thus beware that you will be penalized harder for classification mistakes made on the higher labels.

The score for a random prediction is shown at the end of this notebook (~1.6), your trained models should at least reach a **score below 1** to be significantly better.

## Project Results 

score = 0.57

The success lies in attributing weight during the learning process respectively 1, 10, 100, 1000 to each class. In fact it allocates much weight to small classes and little weight to major class.

## Project pipelines
The project is orchestrated using .yaml file 'config_paths.yaml', there are other way to orchestrate the project using shell files which will be uploaded based on the wrappers.py file and leveraging on argparse python library.

<img align="left" alt="Project pipeline" src="https://camo.githubusercontent.com/1347ad4c15c8a1cb31741627a0bf283fdce8f26847f768d6718ca166922e1621/68747470733a2f2f6c68342e676f6f676c6575736572636f6e74656e742e636f6d2f6e54345467453873354b37642d73476d424b4a455774434a456758495a513439544c56497947365f7430524737506f4e65356b386572414b797337744e6e6f545f7141395f39474859586b4e79773d77313932302d683936352d7277" />

The goal is to propose 3 approaches weither we do have the test dataset or not.
The first one is basically in the case we have the train, test dataset and want to run the full pipeline. 
The second one is in the case we have only the train dataset and want to assess the model quality. 
And the last one is the case we want to solo build a model.

## Project improvements
* Integrate class that stored all transformations done on the train dataset such as the Pipeline functionnality from sklearn package and apply it to any datasets
=> This part will be solved with through the project SKELETON (not updated yet)
* In regards of the size of the data it was not necessary but the next step could be either to reformat the function using mainly numpy instead of pandas, or using the package cudf from RAPIDS
* The current algorithm is based on a weighted lightgbm but we could integrate deep learning framework espacially like CNN ones.



