# privacy-preserving-analytics
Capstone Project @UC Berkeley

## Architecture 
2 folders: 
- archive: files we have shared over the year
- final: essential files for reproducible results

## To get started

1. Download the data generated for this project by downloading the folder 'data' from [here](https://drive.google.com/drive/u/0/folders/1ZrI5eUj9HK4FGaz8ITXussTAIITQy9M3). 

2. Download the ipinyou dataset from [UCL website](http://bunwell.cs.ucl.ac.uk/ipinyou.contest.dataset.zip)

3. Use the files in the quickstart folder to have your first results:
- one-day.ipynb processes 1 day of data to tabular format. (NB: you have to update the first line to link to the location where you put ipinyou.contest.dataset/training2nd from the ipinyou dataset)
- quickstart-fairlearn.ipynb: get quicly started with fairlearn.  

## final
In this folder there are 2 files to reproduce the results we have gathered throughout the year

- bids-to-impression.ipynb: results for the bids to impressions phase. 
- impression-to-clicks.ipynb: results for the impressions to clicks phase. 

In this folder there are also the helper functions (in the folder /helpers) that enabled us to create the datasets for the final results as well as preprocess the data. 


