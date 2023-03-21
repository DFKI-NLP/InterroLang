"""The include operation.

This operation checks whether the input string appears in the dataset.
"""
import string
import re

def includes_operation(conversation, parse_text, i, **kwargs):
    #includes_operation(conversation, text_to_match):
    """The include operation."""
    text_to_match = parse_text[i+1]
    # remove the quotes around the text
    while len(text_to_match)>0 and text_to_match[0] in string.punctuation:
        text_to_match = text_to_match[1:]
    while len(text_to_match)>0 and text_to_match[-1] in string.punctuation:
        text_to_match = text_to_match[:-1]
    text_to_match = re.escape(text_to_match)
    temp_dataset = conversation.temp_dataset.contents["X"]
    text_inputs = temp_dataset["text"]
    
    threshold = 5 # num of words before and after the match
    output_str = ""
    match_len = len(text_to_match)
    for inum, text_input in enumerate(text_inputs):
        matched_indices = [m.start() for m in re.finditer(text_to_match, text_input, re.IGNORECASE)]
        for match_idx in matched_indices:
            before = text_input[:match_idx] 
            match = text_to_match
            after = text_input[match_idx+match_len:]
            before = ' '.join(before.split()[-threshold:])
            after  = ' '.join(after.split()[:threshold])
            idx = temp_dataset.iloc[inum]
            output_str += f"idx {idx}: "
            if len(before)>0:
                output_str+=f"... {before} "
            output_str+=f"<b>{match}</b>"
            if len(after)>0:
                output_str+=f" {after} ..."
            output_str+=" <br>"
    if len(output_str)==0:
        output_str = "No matches were found for: "+text_to_match+"<br>"
    else:
        output_str = f"I found the following matches for <b>{text_to_match}</b>: <br>"+output_str
        
    return output_str, 1


