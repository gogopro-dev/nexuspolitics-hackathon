import json
from collections import defaultdict

import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from data_types import ResponsibleAuthority
from utils import preprocess_text


"""
Compatibility for AMD GPUs on windows, this try/except clause can be removed or commented
out on the linux machines, although it should not be a problem
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
try:
    import torch_directml
    if torch_directml.is_available():
        device = torch_directml.device()
except Exception:
    pass


"""
Load all data about the government entities and the descriptions of categories
"""
categories = json.load(open('data/categories.json', encoding='utf-8'))
entity_catalog = json.load(open('data/entity_catalog.json',  encoding='utf-8'))
entity_phrase_map = dict()
entity_by_state = defaultdict(list)
entity_by_category = defaultdict(list)
bundes_entity_by_category = defaultdict(list)
category_descriptions = categories['descriptions']

for _x in entity_catalog['entities']:
    entity = entity_catalog['entities'][_x]
    keywords = set()
    if entity['state'] is not None:
        entity_by_state[entity['state']].append(entity)



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



embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

label_set = ['Das Problem der Landebene', 'Problem auf Bundesebene']

# Ensemble model names
model_names = [
    "facebook/bart-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "typeform/distilbert-base-uncased-mnli"
]


# Load all models' pipelines
classifiers = [
    pipeline("zero-shot-classification", model=name, device=device)
    for name in model_names
]


# Ensemble prediction function (whether it is State level problem or Local one)
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
# Generate pydocs

def classify_issue(description, category, state) -> ResponsibleAuthority:
    """The function that classifies the issue based on its description, category, and state
    using Zero-Shot ensemble classification and pairwise semantic string matching

    :param description: description of the issue
    :param category: category of the issue
    :param state: state of the issue
    :returns: ResponsibleAuthority
    """

    if description is None:
        raise ValueError("Description cannot be None")
    if category is None:
        raise ValueError("Category cannot be None")
    if state is None:
        raise ValueError("State cannot be None")

    is_state_level = is_state_level_problem_by_description(description)

    # Prefer local institution, if there is none: then add all possible state-level ones
    if not is_state_level:
        possible_entities = [x for x in entity_by_state[state] if x in entity_by_category[category]]

        if len(possible_entities) == 0:
            possible_entities.extend(bundes_entity_by_category[category])

    # Prefer state-level institutions, if there is none: then add all possible local-level ones
    else:
        possible_entities = bundes_entity_by_category[category]
        if len(possible_entities) == 0:
            possible_entities = [x for x in entity_by_state[state] if x in entity_by_category[category]]

    # This should not ever happen
    if len(possible_entities) == 0:
        raise ValueError(f"Possible Entities cannot be zero")

    original = embedder.encode(description, convert_to_tensor=True)
    possible_phrases = [
        x['phrase']
        for x in possible_entities
    ]
    possible_phrases_encoded = embedder.encode(possible_phrases, convert_to_tensor=True)

    cos_scores = util.cos_sim(original, possible_phrases_encoded)[0]
    top_results = torch.topk(cos_scores, k=len(possible_entities))

    # A bit of the counterintuitive
    _id = None
    for score, idx in zip(top_results[0], top_results[1]):
        if _id is None:
            _id = possible_entities[idx]['id']
    return ResponsibleAuthority(
        _id, entity_catalog['entities'][_id]['name'], entity_catalog['entities'][_id]['level']
    )
