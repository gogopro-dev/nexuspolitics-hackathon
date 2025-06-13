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
pip3 install -r requirements.txt
```
2. Run webui (It will take quite a long time on a cold start):
```shell
python3 main.py
```

Now you can open the webui on localhost: http://localhost:7860

# Formal solution as required in the problem statement
The solution from the problem is in the folder `processed_solution`


## Evaluation Metric
Impossible to calculate because of a broken dataset (train, validate) since the authority assigned does not come from 
the federal state defined in the field `state` (explained thoroughly in the presentation), 
in which local authorities are assigned to the issues from wrong regions / topic.

### Credits
Hey, there is no commit history but the thanks should be given where its due, to all people who contributed to this
repository:

`Hlib Zabudko` (me) <br>
`Eriks Spaks` <br>
`Sergey Sarkisyan` <br>
`Nurzhan Zhukesh`

