Hi there,
 
We have three seperate visualisations to show. Unfortunately we could not incorporate them into a single visualisation.
We ran the code using Python 3.10.9.

# Main dashboard
This is the main visualisation of our project and it includes the timeline histogram plot, 
the classifier of the user input, the TSNE and PCA Plot, and the most similar articles based on the user input.

To run this dashboard simply run Dashboard.py in the Main folder and it should work. 
Otherwise you might need to change the in_dir variable where the path to the articles is specified. This variable is specified in prepare_plots.py at line 105 and 192.

prepare_plots.py
This is the main data preprocessing file. Do not run, the dashboard.py file uses the individual functions from this file when loading the data.

# Star graph
PLEASE RUN THE FILES IN SPYDER ENVIRONMENT IF VS CODE GIVES AN ERROR.

new_graph.py
This file produces a star graph that helps us understand how each of the employees are conected to each other and 
what type of emails they exchanged. It also visualises the details of the senders so that the police can check if 
or not that person is suspicious based on their background and line of work. This helps them correlate with the other data visualised.
For this we have preprocessed the data previously to extract important information out of them. 

clean_email.py
This file creates a new file that contains all the senders and receivers having a one-to-one relationship.

Names_pok.py
Goes through the historical documents of Kronos and extracts the names that is mentioned in the report.

Resume.py
Extracts the resumes of all the people in Gastech based on their suspicious background.

# Word cloud
To run this visualisation, simply run word_cloud.py in the word_cloud folder.

The first word cloud creates word cloud for important words in the articles. 
It can be utilized in tandem with the model we made to select articles and look for important words that would pop up. 
The second word cloud shows the important people of POK.