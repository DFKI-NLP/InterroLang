from actions.explanation.topk import topk


def global_top_k(conversation, parse_text, i, **kwargs):
    if "all" in parse_text:
        k = 1
    else:
        for item in parse_text:
            try:
                if int(item):
                    k = int(item)
            except:
                pass

        # Set k 3 by default
        k = 3

    class_name = parse_text[i + 1]

    if class_name == 'true':
        return topk(conversation, "ig_explainer", k,
                    data_path="cache/boolq/ig_explainer_boolq_explanation.json",
                    res_path="cache/boolq/ig_explainer_boolq_attribution.json",
                    print_with_pattern=True, class_name=1), 1
    elif class_name == 'false':
        return topk(conversation, "ig_explainer", k,
                    data_path="cache/boolq/ig_explainer_boolq_explanation.json",
                    res_path="cache/boolq/ig_explainer_boolq_attribution.json",
                    print_with_pattern=True, class_name=0), 1
    else:
        return topk(conversation, "ig_explainer", k,
                    data_path="cache/boolq/ig_explainer_boolq_explanation.json",
                    res_path="./cache/boolq/ig_explainer_boolq_attribution.json",
                    print_with_pattern=True), 1

    # if class_name == "boolq":
    #     return topk("ig_explainer", k,
    #                 data_path="../../cache/boolq/ig_explainer_boolq_explanation.json",
    #                 res_path="../../cache/boolq/ig_explainer_boolq_attribution.json",
    #                 print_with_pattern=True)
    # elif class_name == "dailydialog":
    #     return topk("ig_explainer", k,
    #                 data_path="../../cache/dailydialog/ig_explainer_dailydialog_explanation.json",
    #                 res_path="../../cache/dailydialog/ig_explainer_dailydialog_attribution.json",
    #                 print_with_pattern=True)
    # elif class_name == "olid":
    #     return topk("ig_explainer", k,
    #                 data_path="../../cache/olid/ig_explainer_olid_explanation.json",
    #                 res_path="../../cache/olid/ig_explainer_dailydialog_attribution.json",
    #                 print_with_pattern=True)
    # else:
    #     raise NameError(f"Unknown class name: {class_name}")
