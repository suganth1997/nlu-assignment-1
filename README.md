# Natural Language Understanding: Assignment 1
For training the word vector with vanilla skipgram model use *skipgram.py* </br>
Command line arguments for the script are </br>
```bash
python skipgram.py [word_window_size] [embedding_size] [number_of_epochs]
```
For training the word vectors with negative sampling with default batch size 1 use *negative_sampling.py* </br>
Command line arguments for the script are </br>
```bash
python negative_sampling.py [word_window_size] [embedding_size] [number_of_epochs] [number_of_negative_samples]
```
For training the word vectors with negative sampling with any batch size use *negative_batches.py* </br>
Command line arguments for the script are </br>
```bash
python negative_batches.py [word_window_size] [embedding_size] [number_of_epochs] [number_of_negative_samples] [batch_size]
```
For getting the simlex rating use *testing.py* </br>
Command line argument should be the directory where all the models are pickled *in our case pickled_embeddings* </br>
```bash
python testing_task_1.py [directory_to_pickled_models]
```
To get plots of the results of task 1 use the *plot_results.py* </br>
The script will make use of the result file *correlations.csv*
```bash
python plot_results.py
```
For getting the accuracy for analogy task use *testing_task_2.py* </br>
Command line argument should be the directory to the model embedding that should be tested *will be inside pickled embeddings* </br>
```bash
python testing_task_2.py [directory_to_pickled_model]
```
# Overview of the repository
* pickled_embeddings has all the models </br>
* correlation_scores contains the calculated spearman correlation scores and likelihood probability of validation </br>
* simlex.txt is for the simlex-999 task </br>
* task2.txt is for the analogy task </br>
