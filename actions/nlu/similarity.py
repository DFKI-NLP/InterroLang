import re
from sentence_transformers import SentenceTransformer, util

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

def similar_instances_operation(conversation, parse_text, i, **kwargs):
    """The similarity operation."""
    #similarity {number} {id}
    idx = -1
    number = 3
    return_s = "Sorry, I did not understand the instruction!"
    if len(parse_text)>2:
        if parse_text[i+1].isdigit() and parse_text[i+2].isdigit():
            idx = int(parse_text[i+1])
            number = int(parse_text[i+2])
        # if intent recognition model doesn't find the slot it leaves an empty string
        elif parse_text[i+1].isdigit():
            idx = int(parse_text[i+1])
    if idx>0:
        dataset = conversation.stored_vars["dataset"]
        query = dataset.contents["X"]["text"][idx]
        return_s = get_similar_str(query, idx, number, dataset)
    return return_s, 1


def get_similar_str(query, idx, number, dataset):
    # preparing the output string
    out_str = "The original text for <b>id "+str(idx)+"</b>:<br>"
    query_tokens = query.split()
    query_preview = " ".join(query_tokens[:16])
    out_str+= "<summary>"+query_preview+"...</summary><details>"+query+"</details><br>"
    out_str += "Here are some instances similar to <b>id "+str(idx)+"</b>:<br>"
    found_similars = get_similars(query, idx, dataset, number)
    for cossim, similar_id, similar in found_similars:
        similar_tokens = similar.split()
        similar_preview = " ".join(similar_tokens[:16])
        out_str+="<b>"+str(similar_id)+"</b> (cossim "+str(round(cossim,3))+"):  <summary>"+similar_preview+"...</summary><details>"+similar+"</details><br>"
    return out_str

def get_similars(query, query_idx, dataset, number):
    # computing similarities
    indices = []
    texts = []
    for idx in dataset.contents["index"]:
        if idx!=query_idx:
            indices.append(idx)
            texts.append(dataset.contents["X"]["text"][idx])
    #TA use caching if the dataset is too big?
    query_embedding = similarity_model.encode(query)
    sent_embeddings = similarity_model.encode(texts)
    cos_sim = util.cos_sim(query_embedding, sent_embeddings)[0].tolist()
    # sort by cossim
    similars = []
    for i in range(len(cos_sim)):
        similars.append((cos_sim[i],indices[i],texts[i]))
    similars = sorted(similars, key=lambda x: x[0], reverse=True)
    return similars[:number]

