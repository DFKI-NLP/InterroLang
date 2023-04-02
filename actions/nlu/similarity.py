import re
#import emoji
from sentence_transformers import SentenceTransformer, util


def extract_id_number(parse_text):
    """
    Args:
        parse_text: parsed text from conversation
    Returns:
        id of text and number of cfe instances
    """
    num_list = []
    for item in parse_text:
        try:
            if int(item):
                num_list.append(int(item))
        except:
            pass
    if len(num_list) == 1:
        return num_list[0], 1
    elif len(num_list) == 2:
        return num_list[0], num_list[1]
    else:
        raise ValueError("Too many numbers in parse text!")

def similar_instances_operation(conversation, parse_text, i, **kwargs):

    """
    Args:
        conversation: conversation object
        parse_text: parsed text from conversation
        i: index of operation
    Returns:
        final_results  matched results
    """
    model = conversation.get_var('model').contents
    data = conversation.temp_dataset.contents['X'].values
    labels = conversation.temp_dataset.contents['y'].values
    dataset = data,labels


    if len(conversation.temp_dataset.contents['X']) == 0:
        return 'There are no instances that meet this description!', 0

    id,number = extract_id_number(parse_text)

    final_results = filter_similarity(dataset,parse_text,id,number)

    return final_results


def filter_similarity(dataset, query_sentence, query_label, number):
        """
        Args:
            dataset: dataset from the conversation
            query_sentence: query sentence to be matched
            query_label: label to search for in the dataset
            number: number of hits to be returned
        Returns:
            filtered similarity response to a maximum of specified number
        """

        # if qeury label is 'none' or 'same class'
        results = get_similars(dataset = dataset, query_sentence = query_sentence, query_label = query_label)

        if len(results) == 0:
            return " cannot find any instance",1
        elif number == 1:
            return " ".join(results[0][0]),1
        else:
            if len(results)< number:
                parsed_result = " \n".join([ " ".join(ele[0][0]) for ele in results[:number]])
                return parsed_result,1
            else :
                parsed_result = " \n".join([ " ".join(ele[0][0]) for ele in results])
                return parsed_result,1




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



def get_similars(dataset, query_sentence, query_label):
    """
    Args:
        dataset: dataset from the conversation
        query_sentence: query sentence to be matched
        query_label: label to search for in the dataset
    Returns:
        filtered similarity response
    """

    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('clips/mfaq')
    data, labels = dataset
    # format_training_file('offenseval_train.csv')

    considered_samples = []
    for sample, label in zip(data, labels):
        if str(label) == str(query_label):
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
