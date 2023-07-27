from actions.explanation.topk import topk
from timeout import timeout


@timeout(60)
def global_top_k(conversation, parse_text, i, **kwargs):

    # Set default
    k = 10

    if "all" in parse_text:
        k = 10
    else:
        for item in parse_text:
            try:
                k = int(item)
            except:
                pass

    dataset_name = conversation.describe.get_dataset_name()
    class_names = conversation.class_names
    # Reverse the dictionary
    inverse_class_names = {v.lower(): k for k, v in class_names.items()}

    first_argument = parse_text[i + 1]
    class_idx = None

    if first_argument in list(inverse_class_names.keys()):
        class_idx = inverse_class_names[first_argument]

    reverse = True

    if "least" in parse_text:
        reverse = False

    print(reverse)

    return topk(conversation, "ig_explainer", k,
                data_path=f"./cache/{dataset_name}/ig_explainer_{dataset_name}_explanation.json",
                res_path=f"./cache/{dataset_name}/ig_explainer_{dataset_name}_attribution.json",
                print_with_pattern=True, class_idx=class_idx, reverse=reverse), 1
