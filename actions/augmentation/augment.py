"""Data augmentation operation."""
import string

import nlpaug.augmenter.word as naw

word_aug = naw.ContextualWordEmbsAug(model_path='bert-base-cased', action="substitute")

def augment_operation(conversation, parse_text, i, **kwargs):
    """Data augmentation."""
    model = conversation.get_var('model').contents
    parsed_id = ""
    if len(parse_text)>0:
        parsed_id = parse_text[i+1]
    while parsed_id[-1] in string.punctuation:
        parsed_id = parsed_id[:-1]
    if parsed_id.isdigit():
        id_val = int(parsed_id)
    else:
        return "Sorry, invalid id", 1

    dataset = conversation.get_var('dataset').contents["dataset_name"]
    if dataset=="boolq":
        instance = conversation.get_var('dataset').contents["X"].iloc[id_val]["passage"]
    else:
        instance = conversation.get_var('dataset').contents["X"].iloc[id_val]["text"]

    word_aug_instance = word_aug.augment(instance, n=1)
    
    instance_tok = instance.split()
    word_aug_instance_tok = word_aug_instance.replace("( ","(").replace(" )",")").split()
    word_aug_instance = highlight_changed_tokens(instance_tok, word_aug_instance_tok)

    return_s = ""
    return_s += "<b>Original text:</b><br>"+instance+"</br>"
    return_s += "<b>Augmentation by word replacements:</b><br>"+word_aug_instance+"<br>"
    return return_s, 1

def highlight_changed_tokens(orig_tokens, aug_tokens):
    oi = 0
    ai = 0
    out_str = ""
    while ai < len(aug_tokens):
        if oi>=len(orig_tokens) or orig_tokens[oi]==aug_tokens[ai]:
            out_str+=aug_tokens[ai]+" "
            ai+=1
            oi+=1
        else:
            out_str+="<b><font color='red'>"+aug_tokens[ai]+"</font></b> "
            ai+=1
            while not(ai>=len(aug_tokens) or oi>=len(orig_tokens) or orig_tokens[oi]==aug_tokens[ai]):
                # heuristics to check that oi token appears in aug_tokens
                while (oi<len(orig_tokens) and not(orig_tokens[oi] in aug_tokens[ai:ai+3])):
                    oi+=1
                if oi>=len(orig_tokens):
                    break
                elif orig_tokens[oi]==aug_tokens[ai]:
                    out_str+=aug_tokens[ai]+" "
                    ai+=1
                    oi+=1
                    break
                out_str+="<b><font color='red'>"+aug_tokens[ai]+"</font></b> "
                ai+=1
    return out_str
    
