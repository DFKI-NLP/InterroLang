"""Data summary operation."""
import nltk


def get_frequent_words(conversation, f_names, top=5):
    df = conversation.temp_dataset.contents['X']

    nltk.download("stopwords")
    sw = nltk.corpus.stopwords.words('english')
    temp = ""
    for f in f_names:
        for i in range(len(df[f])):
            temp += df[f][i]
            temp += " "

    words = temp.split(" ")

    words_ne = []
    for word in words:
        if word not in sw:
            words_ne.append(word)

    word_dict = dict(nltk.FreqDist(words_ne))

    frequent_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:top]

    return frequent_words


# Note, these are hardcode for compas!
def data_operation(conversation, parse_text, i, **kwargs):
    """Data summary operation."""
    description = conversation.describe.get_dataset_description()
    text = f"The data contains information related to <b>{description}</b>.<br>"

    # List out the feature names
    f_names = list(conversation.temp_dataset.contents['X'].columns)

    # return str(f_names), 1
    f_string = "<ul>"
    for fn in f_names:
        f_string += f"<li>{fn}</li>"
    f_string += "</ul>"
    text += f"The exact <b>feature names</b> in the data are listed as follows:{f_string}<br><br>"

    class_list = list(conversation.class_names.values())
    text += "The dataset has following <b>labels</b>: "
    text += "<ul>"
    for i in range(len(class_list)):
        text += "<li>"
        text += str(class_list[i])
        text += "</li>"
    text += "</ul><br><br>"

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

    frequent_words = get_frequent_words(conversation, f_names, top=top)

    text += "<table>"
    text += "<tr>"
    text += "<th> Word </th>"
    text += "<th> Frequence </th>"
    text += "</tr>"

    for i in range(len(frequent_words)):
        text += "<tr>"
        text += f"<td> {frequent_words[i][0]} </td>"
        text += f"<td> {frequent_words[i][1]} </td>"
        text += "</tr>"
    text += "</table><br><br>"

    # for i, f in enumerate(f_names):
    #     mean = round(df[f].mean(), conversation.rounding_precision)
    #     std = round(df[f].std(), conversation.rounding_precision)
    #     min_v = round(df[f].min(), conversation.rounding_precision)
    #     max_v = round(df[f].max(), conversation.rounding_precision)
    #     new_feature = (f"{f}: The mean is {mean}, one standard deviation is {std},"
    #                    f" the minimum value is {min_v}, and the maximum value is {max_v}")
    #     new_feature += "<br><br>"
    #
    #     rest_of_text += new_feature
    #
    # text += "Let me know if you want to see an in depth description of the dataset statistics.<br><br>"
    # conversation.store_followup_desc(rest_of_text)

    return text, 1
