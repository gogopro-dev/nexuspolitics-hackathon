# Automatic identification of responsible authorities for citizen appeals

## 📌 Project Description
This system automatically determines responsible government entities for citizen complaints by analyzing problem 
descriptions, categories, and geographic locations to identify the appropriate authority (federal or state level).

## Contents
This project contains a simple webserver with a http api that can assign issues to authorities based on the problem
region, description and category. 

There is also a utility in `csv_processor` directory that applies the API method localy to a csv file from the example.
The guide for using it is located in the directory in the `csv_preprocessor/readme.md`

## Algorithm Overview
**Hybrid approach combining:**
1. **Rule-based matching**:
   - Category-to-entity competency mapping
   - Geographic filtering (state/region)
2. **Semantic text analysis**:
   - Vector embeddings of texts and competencies
   - Cosine similarity calculations of pairwise semantic 



### Installation and Usage of the webserver

1. Install Dependencies (it is advised to use a virtual environment)
```shell
pip3 intsall -r requirements.txt
```
2. Run webui (It will take quite a long time on a cold start):
```shell
python3 main.py
```

Now you can open the webui on localhost: http://localhost:7860
