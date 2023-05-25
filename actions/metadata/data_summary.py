"""Data summary operation."""
import nltk


def get_frequent_words(conversation, f_names, top=5):
    """

    Args:
        conversation: conversation object
        f_names: list of feature names
        top: top k frequent words

    Returns:
        frequent_words: list of tuples in form: (word, freq)
    """
    df = conversation.temp_dataset.contents["X"]

    nltk.download("stopwords")
    sw = nltk.corpus.stopwords.words("english")
    temp = ""
    for f in f_names:
        for inum, t in enumerate(df[f]):
            temp += str(t) + " "

    words = temp.split(" ")

    words_ne = []
    for word in words:
        if word not in sw:
            words_ne.append(word)

    word_dict = dict(nltk.FreqDist(words_ne))

    frequent_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:top]

    text = "<table style='border: 1px solid black;'>"
    text += "<tr style='border: 1px solid black;'>"
    text += "<th> Word </th>"
    text += "<th> Frequence </th>"
    text += "</tr>"

    for i in range(len(frequent_words)):
        text += "<tr style='border: 1px solid black;'>"
        text += f"<td style='border: 1px solid black;'> {frequent_words[i][0]} </td>"
        text += f"<td style='border: 1px solid black;'> {frequent_words[i][1]} </td>"
        text += "</tr>"
    text += "</table><br>"

    return text


def keyword_operation(conversation, parse_text, i, **kwargs):
    """topk keywords operation. """
    df = conversation.temp_dataset.contents["X"]

    # List out the feature names
    f_names = list(df.columns)

    # Extract topk value
    if "keywords all" in " ".join(parse_text):
        top = 25
    else:
        num_list = []
        for item in parse_text:
            try:
                if int(item):
                    num_list.append(int(item))
            except ValueError:
                pass
        top = num_list[-1]

    return_s = f"The {top} most frequent words in the dataset are: <br>"
    return_s += get_frequent_words(conversation, f_names=f_names, top=top)

    return return_s, 1


def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    description = conversation.describe.get_dataset_description()
    text = f"The data contains information related to <b>{description}</b>.<br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)

    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    text += f"The exact <b>feature names</b> in the data are listed as follows:{f_string}<br>"

    class_list = list(conversation.class_names.values())
    text += "The dataset has following <b>labels</b>: "
    text += "<ul>"
    for i in range(len(class_list)):
        text += "<li>"
        text += str(class_list[i])
        text += "</li>"
    text += "</ul><br>"

    # Summarize performance
    dataset_name = conversation.describe.get_dataset_name()
    score = conversation.describe.get_eval_performance_for_hf_model(dataset_name, conversation.default_metric)

    # Note, if no eval data is specified this will return an empty string and nothing will happen.
    if score != "":
        text += score
        text += "<br><br>"

    # Create more in depth description of the data, summarizing a few statistics
    top = 5
    text += f"Here's a more in depth summary of the data. The topk {top} most frequent words among the dataset are: "
    text += "<br><br>"

    text += get_frequent_words(conversation, f_names, top=top)

    return text, 1
