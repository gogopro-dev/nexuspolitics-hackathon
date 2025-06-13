import torch
import pandas as pd
import json
import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from collections import defaultdict

tqdm.pandas()
nltk.download('stopwords')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
# Compatibility for AMD GPUs on windows, this try/except clause can be removed or commented
# out on the linux machines, although it should not be a problem
try:
    import torch_directml
    if torch_directml.is_available() and device == torch.device("cpu"):
        device = torch_directml.device()
except Exception:
    pass




df = pd.read_csv('../data/in.csv', on_bad_lines='skip')  # Or ISO-8859-1 or whatever works
categories = json.load(open('../data/categories.json', encoding='utf-8'))
entity_catalog = json.load(open('../data/entity_catalog.json',  encoding='utf-8'))
entity_phrase_map = dict()
entity_by_region = defaultdict(list)
entity_by_category = defaultdict(list)
bundes_entity_by_category = defaultdict(list)
category_descriptions = categories['descriptions']

def preprocess_text(text: str) -> str:
    # remove html tags
    text = re.sub(r"<[^>]*>", " ", text)
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^a-zA-Z0-9äöüÄÖÜß ]+", " ", text)

    # remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("german")]
    text = " ".join(tokens)
    text = text.lower().strip()

    return text


for _x in entity_catalog['entities']:
    entity = entity_catalog['entities'][_x]
    keywords = set()
    if entity['state'] is not None:
        entity_by_region[entity['state']].append(entity)



    for x in entity['competencies']:
        keywords.add(preprocess_text(x))

    for x in entity['keywords']:
        keywords.add(preprocess_text(x))

    entity_phrase_map[entity['id']] = ' '.join(list(keywords))
    entity['phrase'] = entity_phrase_map[entity['id']]

for _category_map in entity_catalog['category_entity_map']:
    entities = entity_catalog['category_entity_map'][_category_map]
    category = _category_map
    for entity in entities:
        entity_data = entity_catalog['entities'][entity]
        if entity_data['level'] == 'Bund':
            bundes_entity_by_category[category].append(entity_data)
        entity_by_category[category].append(entity_data)




df['description'] = df['description'].progress_apply(lambda _var: preprocess_text(_var))

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


label_set = ['Das Problem der Landebene', 'Problem auf Bundesebene']
#
## Define ensemble model names
model_names = [
    "facebook/bart-large-mnli",

    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",

    "typeform/distilbert-base-uncased-mnli"
]
#
# Load all pipelines
classifiers = [
    pipeline("zero-shot-classification", model=name, device=device)
    for name in model_names
]


# Ensemble prediction function
def is_state_level_problem_by_description(description: str) -> bool:
    all_scores = []
    for pline in classifiers:
        result = pline(description, candidate_labels=label_set)
        # Match score order to label_set
        scores = [result["scores"][result["labels"].index(lbl)] for lbl in label_set]
        all_scores.append(scores)
    avg_scores = np.mean(all_scores, axis=0)
    best_idx = np.argmax(avg_scores)
    return label_set.index(label_set[best_idx]) == 1


outdf = pd.DataFrame(columns=['issue_id', 'responsible_entity_id'])
total_interation_done = 0

# Process records one-by-one
for index, row in df.iterrows():
    region = row['state']
    total_interation_done += 1
    print(f"total_interation_done: {total_interation_done}")
    is_state_level = is_state_level_problem_by_description(row['description'])

    # Prefer local institution, if there is none: then add all possible state-level ones
    if not is_state_level:
        possible_entities = [x for x in entity_by_region[region] if x in entity_by_category[row['category']]]

        if len(possible_entities) == 0:
            possible_entities.extend(bundes_entity_by_category[row['category']])

    # Prefer state-level institutions, if there is none: then add all possible local-level ones
    else:
        possible_entities = bundes_entity_by_category[row['category']]
        if len(possible_entities) == 0:
            possible_entities = [x for x in entity_by_region[region] if x in entity_by_category[row['category']]]

    if len(possible_entities) == 0:
        raise ValueError(f"Possible Entities cannot be zero, index: {index}")

    original = embedder.encode(row['description'],  convert_to_tensor=True)
    possible_phrases = [
        x['phrase']
        for x in possible_entities
    ]
    possible_phrases_encoded = embedder.encode(possible_phrases, convert_to_tensor=True)

    cos_scores = util.cos_sim(original, possible_phrases_encoded)[0]
    top_results = torch.topk(cos_scores, k=len(possible_entities))

    print(f"\n\n===========  <{index}>  ===========\n")
    print(row['issue_id'])
    print(row['description'])
    _id = None
    for score, idx in zip(top_results[0], top_results[1]):
        if _id is None:
            _id = possible_entities[idx]['id']
        print(possible_entities[idx]['id'], possible_entities[idx]['name'], "(Score: {:.4f})".format(score), possible_phrases[idx], end='\n\n')
    outdf.loc[index] = [row['issue_id'], _id]


outdf.to_csv('out.csv', index=False)