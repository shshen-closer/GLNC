# Source code and data set for the paper: Global and Local Cognitive Modeling for Student Performance Prediction.




## Dependencies:

- python >= 3.7
- tesorflow-gpu >= 2.0 
- numpy
- tqdm
- utils
- pandas
- sklearn


## Usage

First, download the data file, then put it in the folder 'data/' 

Then, run data_pre.py to preprocess the data set, and run data_save.py {sequence length} to divide the original data set into train set, validation set and test set. 

`python generate_data.py {fold}`

For example:

`python generate_data.py 1`  or `python generate_data.py 2`

Train the model:

`python train.py 30`



