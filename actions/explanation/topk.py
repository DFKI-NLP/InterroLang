import torch
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm

# attribution_to_html is based on this tutorial:
# https://github.com/CVxTz/interpretable_nlp
from colour import Color
COLOR_RANGE = list(Color("red").range_to(Color("white"), 10)) + list(
    Color("white").range_to(Color("green"), 10))

def attribution_to_html(tokens, attributions):
    html = ""
    for token, attribution in zip(tokens, attributions):
        if attribution >= 0:
            idx = int(attribution ** 1 * 10) + 10
        else:
            idx = int((-(-attribution) ** 1 + 1) * 10)
        idx = min(idx, 19)
        # making it colourful based on the attributions
        color = COLOR_RANGE[idx]
        if token == "[PAD]":
            continue
        if token.startswith("##"):
            token = token.replace("##","")
            html += f"""<span style="background-color: {color.hex}">{token}</span>"""
        else:
            html += f""" <span style="background-color: {color.hex}">{token}</span>"""
    return html 


class FeatureAttributionExplainer():
    def __init__(self, model, dataloader, tokenizer, dataset_name):
        super(FeatureAttributionExplainer).__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        self.forward_func = self.get_forward_func()
        self.explainer = LayerIntegratedGradients(forward_func=self.forward_func,
                                                  layer=self.get_embedding_layer())
        self.pad_token_id = tokenizer.pad_token_id

    def get_forward_func(self):
        def bert_forward(input_ids, attention_mask):
            input_model = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
            }
            output_model = self.model(input_ids=input_ids, attention_mask = attention_mask).logits
            return output_model

        return bert_forward

    def get_embedding_layer(self):
        return self.model.base_model.embeddings
        
    def get_inputs_and_additional_args(self, base_model, batch):
        assert 'input_ids' in batch, f'Input ids expected for {base_model} but not found.'
        assert 'attention_mask' in batch, f'Attention mask expected for {base_model} but not found.'
        input_ids = batch['input_ids']
        additional_forward_args = (batch['attention_mask'])
        return input_ids, additional_forward_args
  
    def get_baseline(self, batch):
        assert 'special_tokens_mask' in batch
        if self.pad_token_id == 0:
            # all non-special token ids are replaced by 0, the pad id
            baseline = batch['input_ids'] * batch['special_tokens_mask']
            return baseline
        else:
            baseline = batch['input_ids'] * batch['special_tokens_mask']  # all input ids now 0
            # add pad_id everywhere,
            # substract again where special tokens are, leaves non special tokens with pad id
            # and conserves original pad ids
            baseline = (baseline + self.pad_token_id) - (batch['special_tokens_mask'] * self.pad_token_id)
            return baseline

    def compute_feature_attribution_scores(
            self,
            batch
    ):
        r"""
        :param batch
        :return:
        """
        self.model.eval()
        self.model.zero_grad()

        batch = {k: v.to(self.device) for k, v in batch.items()}
        inputs, additional_forward_args = self.get_inputs_and_additional_args(
            base_model=type(self.model),
            batch=batch
        )

        predictions = self.forward_func(
            inputs,
            additional_forward_args
        )
        pred_id = torch.argmax(predictions, dim=1)

        baseline = self.get_baseline(batch=batch)

        
        attributions = self.explainer.attribute(
            inputs=inputs,
            n_steps=50,
            additional_forward_args=additional_forward_args,
            target=pred_id,
            baselines=baseline,
            internal_batch_size=1,
        )
        attributions = torch.sum(attributions, dim=2)
        
        return attributions, predictions
        
    def generate_explanation(self):
        def detach_to_list(t):
            return t.detach().cpu().numpy().tolist() if type(t) == torch.Tensor else t

        all_predictions = []
        all_scores = []
        all_html = []
        for idx_batch, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), position=0, leave=True):
            
            batch_labels = batch["label"]
            # check which dataset we use
            input_list = []
            if self.dataset_name=="boolq":
                input_list = [(batch["passage"][i], batch["question"][i]) for i in range(len(batch["passage"]))]
            elif self.dataset_name=="dailydialog":
                input_list = batch["text"]

            batch = self.tokenizer.batch_encode_plus(input_list,
                         truncation="longest_first",
                         padding="max_length",
                         max_length=256, return_special_tokens_mask=True, return_tensors='pt')
            attribution, predictions = self.compute_feature_attribution_scores(batch)

            for idx_instance in range(len(batch['input_ids'])):
                idx_instance_running = (idx_batch * self.dataloader.batch_size)

                ids = detach_to_list(batch['input_ids'][idx_instance])
                label = batch_labels[idx_instance]                
                preds = detach_to_list(predictions[idx_instance])

                attributions_ig = attribution[idx_instance]
                attr = detach_to_list(attributions_ig)
                # normalize the scores for display         
                attributions_ig = attributions_ig / max([abs(el) for el in attributions_ig])
                tokens =  self.tokenizer.tokenize(self.tokenizer.decode(ids))
                # sort the tokens by their absolute attribution values
                abs_attr = [abs(el) for el in attr]
                scores = zip(tokens, attr, abs_attr)
                sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
                # remove the abs scores from the final list
                sorted_scores = [(el[0], el[1]) for el in sorted_scores]
                # create visualization
                html = attribution_to_html(tokens, attributions_ig)
                model_prediction = torch.argmax(predictions[idx_instance]).item()
                all_predictions.append(model_prediction)
                all_scores.append(sorted_scores)
                all_html.append(html)
                
            return all_predictions, all_scores, all_html



def topk_operation(conversation, parse_text, i, **kwargs):
    return_s = ""
    if len(parse_text[i+1])>0:
        k_val = int(parse_text[i+1])
    else:
        k_val = 10
    if len(parse_text[i+2])>0:
        id_val = int(parse_text[i+2])
    else:
        id_val = -1
    if id_val<0:
        return "Invalid id, please try again!", 1
    # load the model, tokenizer etc.
    transformer_model = conversation.get_var("model").contents
    model = transformer_model.model #...conversation.add_var('model', model, 'model')
    tokenizer = transformer_model.tokenizer
    id2label = conversation.get_var('dataset').contents["id2label"]
    dataset_name = conversation.get_var('dataset').contents["dataset_name"]
    x_data = conversation.get_var('dataset').contents["X"]
    # prepare the dataset
    if dataset_name=="boolq":
        instance_passage = x_data.iloc[id_val]["passage"]
        instance_question = x_data.iloc[id_val]["question"]
        instance_label = conversation.get_var('dataset').contents["y"][id_val]#.iloc[id_val]
        dataset = [{"question": instance_question, "passage":instance_passage, "label": instance_label}]
    elif dataset_name=="dailydialog":
        instance_texts = [ast.literal_eval(txt) for txt in x_data.iloc[id_val]["dialog"]] 
        instance_labels = conversation.get_var('dataset').contents["y"][id_val]#.iloc[id_val]
        dataset = [{"text": instance_texts, "label": instance_labels}]
    # get attributions for each batch    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    all_model_predictions, all_sorted_scores, all_html = FeatureAttributionExplainer(model, dataloader, tokenizer, dataset_name).generate_explanation()
    # create the output string
    for j in range(len(all_model_predictions)):
        model_prediction = all_model_predictions[j]
        model_prediction = id2label[model_prediction]
        sorted_scores = all_sorted_scores[j]
        html = all_html[j]
        return_s += "<b>Id: </b>" + str(id_val) + "<br>"
        if dataset_name=="boolq":
            return_s += "<b>Passage:</b><br>" + instance_passage + "<br>"
            return_s += "<b>Question:</b><br>" + instance_question + "<br>"
        elif dataset_name=="dailydialog":
            return_s += "<b>Dialog:</b><br>" + instance_texts[j] + "<br>"

        return_s += "<b>Prediction:</b> " + model_prediction + "<br>"
        top_tokens_str = ""
        for token, score in sorted_scores[:k_val]:
            top_tokens_str+=token + ": " + str(round(score,3))+"<br>"
        return_s += "<br>"    
        return_s += "<b>Top "+ str(k_val) +"</b> contributing tokens: <br>"+top_tokens_str+"<br>"+html+"<br>"
    return return_s, 1

