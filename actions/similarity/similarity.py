import re
#import emoji
from sentence_transformers import SentenceTransformer, util


def format_training_file(text_file, module_path=''):
    tweets = []
    classes = []

    for line in open(module_path+text_file,'r',encoding='utf-8'):
        line = re.sub(r'#([^ ]*)', r'\1', line)
        line = re.sub(r'https.*[^ ]', 'URL', line)
        line = re.sub(r'http.*[^ ]', 'URL', line)
        #line = emoji.demojize(line)
        line = re.sub(r'(:.*?:)', r' \1 ', line)
        line = re.sub(' +', ' ', line)
        line = line.rstrip('\n').split(',')
        tweets.append(' '.join(line[:-1]))
        classes.append(line[-1])

    return tweets[1:], classes[1:]



def get_similars(query_sentence, query_label):
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('clips/mfaq')
    data, labels = format_training_file('offenseval_train.csv')

    considered_samples = []
    for sample, label in zip(data, labels):
        if label == str(query_label):
            considered_samples.append(sample)



    query_embedding = model.encode(query_sentence)
    sample_embeddings = model.encode(considered_samples)

    similar_pairs = util.semantic_search(query_embedding, sample_embeddings)
    pairs = similar_pairs[0]

    similars = []
    for pair in pairs:
        similar = considered_samples[pair['corpus_id']]
        score = pair['score']
        similars.append((similar, score))
    return similars

def main():
    query_sentence = 'StopKavanaugh he is liar like the rest of the GOP URL'
    
    query_label = 0
    number = 1
    if query_label == 1:
        # if qeury label is 'none' or 'same class'
        results = get_similars(query_sentence=query_sentence, query_label=1)
        if number == 1:
            # a similar sentence
            # a similar input
            # a similar sample
            # a similar example
            # an equivalent example
            # an equivalent input
            # most similar example
            # one similar example
            return results[0]
        else:
            # some similar examples
            # a few similar example
            # some similar examples
            # a few similar examples
            # similar examples
            return results[:3]

    
    else:
        # if qeury label is 'other class, oposite class, etc.'
        results = get_similars(query_sentence=query_sentence, query_label=0)
        if number == 1:
            # a similar sentence
            # a similar input
            # a similar sample
            # a similar example
            # an equivalent example
            # an equivalent input
            # most similar example
            # one similar example
            return results[0]
        else:
            # some similar examples
            # a few similar example
            # some similar examples
            # a few similar examples
            # similar examples
            return results[:3]

    
 


if __name__ == '__main__':
    x = main()
    print(x)
