"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""

import gin
import numpy as np
import os
import re
import pickle
import secrets
import sys
import torch
from flask import Flask
from random import seed as py_random_seed

from word2number import w2n
import string

from logic.action import run_action
from logic.conversation import Conversation
from logic.decoder import Decoder
from logic.parser import Parser, get_parse_tree
from logic.prompts import Prompts
from logic.utils import read_and_format_data
from logic.write_to_log import log_dialogue_input

from transformers import AutoAdapterModel, AutoTokenizer
from transformers import TextClassificationPipeline, TokenClassificationPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@gin.configurable
def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


from logic.transformers import TransformerModel


@gin.configurable
def load_hf_model(model_id):
    """ Loads a (local) Hugging Face model from a directory containing a pytorch_model.bin file and a config.json file.
    """
    return TransformerModel(model_id)
    # transformers.AutoModel.from_pretrained(model_id)


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: list[str],
                 numerical_features: list[str],
                 remove_underscores: bool,
                 name: str,
                 parsing_model_name: str = "ucinlp/diabetes-t5-small",
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 t5_config: str = None,
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False):
        """The init routine.

        Arguments:
            model_file_path: The filepath of the **user provided** model to logic. This model
                             should end with .pkl and support sklearn style functions like
                             .predict(...) and .predict_proba(...)
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
            background_dataset_file_path: The path to the dataset used for the 'background' data
                                          in the explanations.
            dataset_index_column: The index column in the data. This is used when calling
                                  pd.read_csv(..., index_col=dataset_index_column)
            target_variable_name: The name of the column in the dataset corresponding to the target,
                                  i.e., 'y'
            categorical_features: The names of the categorical features in the data. If None, they
                                  will be guessed.
            numerical_features: The names of the numeric features in the data. If None, they will
                                be guessed.
            remove_underscores: Whether to remove underscores in the feature names. This might help
                                performance a bit.
            name: The dataset name
            parsing_model_name: The name of the parsing model. See decoder.py for more details about
                                the allowed models.
            seed: The seed
            prompt_metric: The metric used to compute the nearest neighbor prompts. The supported options
                           are cosine, euclidean, and random
            prompt_ordering:
            t5_config: The path to the configuration file for t5 models, if using one of these.
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
        """
        super(ExplainBot, self).__init__()
        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)
        torch.manual_seed(seed)

        self.bot_name = name

        # Prompt settings
        self.prompt_metric = prompt_metric
        self.prompt_ordering = prompt_ordering
        self.use_guided_decoding = use_guided_decoding

        # A variable used to help file uploads
        self.manual_var_filename = None

        self.decoding_model_name = parsing_model_name

        # Initialize completion + parsing modules
        app.logger.info(f"Loading parsing model {parsing_model_name}...")
        self.decoder = Decoder(parsing_model_name,
                               t5_config,
                               use_guided_decoding=self.use_guided_decoding,
                               dataset_name=name)

        # Initialize parser + prompts as None
        # These are done when the dataset is loaded
        self.prompts = None
        self.parser = None

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions)

        # Load the model into the conversation
        self.load_model(model_file_path)

        # Load the dataset into the conversation
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores,
                          store_to_conversation=True,
                          skip_prompts=skip_prompts)
                          
        self.adapter_model = AutoAdapterModel.from_pretrained("bert-base-cased")
        self.adapter_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.include_adapter = self.adapter_model.load_adapter("./intents_and_slots/intent_slot_classification/adapters/include")
        self.nlpcfe_adapter = self.adapter_model.load_adapter("./intents_and_slots/intent_slot_classification/adapters/nlpcfe")
        self.similar_adapter = self.adapter_model.load_adapter("./intents_and_slots/intent_slot_classification/adapters/similar")
        self.predict_adapter = self.adapter_model.load_adapter("./intents_and_slots/intent_slot_classification/adapters/predict")
        self.adapters = {
            "include":self.include_adapter,
            "nlpcfe":self.nlpcfe_adapter,
            "similar":self.similar_adapter,
            "predict":self.predict_adapter,
        }

        self.quote_pattern = r'(\"|\')[^\"\']*(\"|\')'

        self.tagger_model = AutoAdapterModel.from_pretrained("bert-base-cased")
        includetoken_path = "./intents_and_slots/intent_slot_classification/adapters/includetoken"
        self.includetoken_adapter = self.tagger_model.load_adapter(includetoken_path)
        number_path = "./intents_and_slots/intent_slot_classification/adapters/number"
        self.number_adapter = self.tagger_model.load_adapter(number_path)
        id_path = "./intents_and_slots/intent_slot_classification/adapters/id"
        self.id_adapter = self.tagger_model.load_adapter(id_path)
        # load heads
        self.tagger_model.load_head(includetoken_path)
        self.tagger_model.load_head(number_path)
        self.tagger_model.load_head(id_path)

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def load_model(self, filepath: str):
        """Loads a model.

        This routine loads a model into the conversation
        from a specified file path. The model will be saved as a variable
        names 'model' in the conversation, overwriting an existing model.

        The routine determines the type of model from the file extension.
        Scikit learn models should be saved as .pkl's and torch as .pt.

        Arguments:
            filepath: the filepath of the model.
        Returns:
            success: whether the model was saved successfully.
        """
        # app.logger.info(f"Loading inference model at path {filepath}...")
        # if filepath.endswith('.pkl'):
        #     model = load_sklearn_model(filepath)
        #     self.conversation.add_var('model', model, 'model')
        #     self.conversation.add_var('model_prob_predict',
        #                               model.predict_proba,
        #                               'prediction_function')
        # else:
        #     # No other types of models implemented yet
        #     message = (f"Models with file extension {filepath} are not supported."
        #                " You must provide a model stored in a .pkl that can be loaded"
        #                f" and called like an sklearn model.")
        #     raise NameError(message)
        # app.logger.info("...done")
        # return 'success'
        app.logger.info(f"Loading inference model at path {filepath}...")
        if filepath.endswith('.pkl'):
            model = load_sklearn_model(filepath)
            self.conversation.add_var('model', model, 'model')
            self.conversation.add_var('model_prob_predict',
                                      model.predict_proba,
                                      'prediction_function')
        else:
            model = load_hf_model(filepath)
            self.conversation.add_var('model', model, 'model')
            # self.conversation.add_var('model_prob_predict', model, 'prediction_function')
        """
        else:
            # No other types of models implemented yet
            message = (f"Models with file extension {filepath} are not supported."
                       " You must provide a model stored in a .pkl that can be loaded"
                       f" and called like an sklearn model.")
            raise NameError(message)
        """
        app.logger.info("...done")
        return 'success'

    def load_dataset(self,
                     filepath: str,
                     index_col: int,
                     target_var_name: str,
                     cat_features: list[str],
                     num_features: list[str],
                     remove_underscores: bool,
                     store_to_conversation: bool,
                     skip_prompts: bool = False):
        """Loads a dataset, creating parser and prompts.

        This routine loads a dataset. From this dataset, the parser
        is created, using the feature names, feature values to create
        the grammar used by the parser. It also generates prompts for
        this particular dataset, to be used when determine outputs
        from the model.

        Arguments:
            filepath: The filepath of the dataset.
            index_col: The index column in the dataset
            target_var_name: The target column in the data, i.e., 'y' for instance
            cat_features: The categorical features in the data
            num_features: The numeric features in the data
            remove_underscores: Whether to remove underscores from feature names
            store_to_conversation: Whether to store the dataset to the conversation.
            skip_prompts: whether to skip prompt generation.
        Returns:
            success: Returns success if completed and store_to_conversation is set to true. Otherwise,
                     returns the dataset.
        """
        app.logger.info(f"Loading dataset at path {filepath}...")

        # Read the dataset and get categorical and numerical features
        dataset, y_values, categorical, numeric = read_and_format_data(filepath,
                                                                       index_col,
                                                                       target_var_name,
                                                                       cat_features,
                                                                       num_features,
                                                                       remove_underscores)

        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)

            # Set up the parser
            self.parser = Parser(cat_features=categorical,
                                 num_features=numeric,
                                 dataset=dataset,
                                 target=list(y_values))

            # Generate the available prompts
            # make sure to add the "incorrect" temporary feature
            # so we generate prompts for this
            self.prompts = Prompts(cat_features=categorical,
                                   num_features=numeric,
                                   target=np.unique(list(y_values)),
                                   feature_value_dict=self.parser.features,
                                   class_names=self.conversation.class_names,
                                   skip_creating_prompts=skip_prompts)
            app.logger.info("..done")

            return "success"
        else:
            return dataset

    def set_num_prompts(self, num_prompts):
        """Updates the number of prompts to a new number"""
        self.prompts.set_num_prompts(num_prompts)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """To uniquely identify each input, we generate a random 30 byte hex string."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Performs the system logging."""
        assert isinstance(logging_input, dict), "Logging input must be dict"
        assert "time" not in logging_input, "Time field will be added to logging input"
        log_dialogue_input(logging_input)

    @staticmethod
    def build_logging_info(bot_name: str,
                           username: str,
                           response_id: str,
                           system_input: str,
                           parsed_text: str,
                           system_response: str):
        """Builds the logging dictionary."""
        return {
            'bot_name': bot_name,
            'username': username,
            'id': response_id,
            'system_input': system_input,
            'parsed_text': parsed_text,
            'system_response': system_response
        }
        
    def check_heuristics(self, decoded_text: str, orig_text: str):
        """Checks heuristics for those intents/actions that were identified but their core slots are missing.
        """
        if "includes" in decoded_text and "{includetoken}" in decoded_text:
            indicators = ["word ", "words ", "token ", "tokens "]
            for indicator in indicators:
                if indicator in orig_text:
                    word_start = orig_text.index(indicator)+len(indicator)
                    if word_start<len(orig_text):
                        includeword = orig_text[word_start:]
                        decoded_text = decoded_text.replace("{includetoken}", includeword)
                        break
            # check for quotes
            if "{includetoken}" in decoded_text:
                in_quote = re.search(self.quote_pattern, orig_text)
                if  in_quote is not None:
                   decoded_text = decoded_text.replace("{includetoken}", in_quote.group()) 
        if "{id}" in decoded_text:
            if "id" in text:
                splitted = text[text.index("id ")+2:].strip().split()
                if len(splitted)>0:
                    decoded_text = decoded_text.replace("{id}", splitted[0])
        return decoded_text

    def get_num_value(self, text: str):
        """Converts text to number if possible"""
        for ch in string.punctuation:
            if ch in text:
                text = text.replace(ch,"")
        if len(text)>0 and not(text.isdigit()):
            converted_num = w2n.word_to_num(text)
            if converted_num is not None:
                text = str(converted_num)
        return text


    def compute_parse_text(self, text: str, error_analysis: bool = False):
        """Computes the parsed text from the user text input.

        Arguments:
            error_analysis: Whether to do an error analysis step, where we compute if the
                            chosen prompts include all the
            text: The text the user provides to the system
        Returns:
            parse_tree: The parse tree from the formal grammar decoded from the user input.
            parse_text: The decoded text in the formal grammar decoded from the user input
                        (Note, this is just the tree in a string representation).
        """
        nn_prompts = None
        if error_analysis:
            grammar, prompted_text, nn_prompts = self.compute_grammar(text, error_analysis=error_analysis)
        else:
            grammar, prompted_text = self.compute_grammar(text, error_analysis=error_analysis)
        app.logger.info("About to decode")
        # Do guided-decoding to get the decoded text
        api_response = self.decoder.complete(
            prompted_text, grammar=grammar)
        decoded_text = api_response['generation']
        decoded_text = self.check_heuristics(decoded_text, text)

        app.logger.info(f'Decoded text {decoded_text}')

        # Compute the parse tree from the decoded text
        # NOTE: currently, we're using only the decoded text and not the full
        # tree. If we need to support more complicated parses, we can change this.
        parse_tree, parsed_text = get_parse_tree(decoded_text)
        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

    def compute_parse_text_t5(self, text: str):
        """Computes the parsed text for the input using a t5 model.

        This supposes the user has finetuned a t5 model on their particular task and there isn't
        a need to do few shot
        """
        grammar, prompted_text = self.compute_grammar(text)
        decoded_text = self.decoder.complete(text, grammar)
        decoded_text = self.check_heuristics(decoded_text, text)
        app.logger.info(f"t5 decoded text {decoded_text}")
        parse_tree, parse_text = get_parse_tree(decoded_text[0])
        return parse_tree, parse_text

    def switch_case(self, best_intent):
        switch = { "include": "includes {includetoken} [e]",
                   "nlpcfe":  "nlpcfe {number} {id} [e]", 
                   "predict": "predict {id} [e]", 
                   "similar": "similar {number} {id} [e]",
                 }
        return switch.get(best_intent, "")

    def get_tagged_span(self, tagged: list[dict], text: str):
        """Extracts the longest contiguous span from the tagger.
        """
        if len(tagged)>0:
            start_span = None
            end_span = None
            for el in tagged:
                start = el['start']
                end = el['end']
                if start_span is None or start<start_span:
                    start_span = start
                if end_span is None or end>end_span:    
                    end_span = end
            token_string = text[start_span:end_span]
        return token_string

    def compute_parse_text_adapters(self, text: str):
        """Computes the parsed text for the input using intent classifier model.
        """
        parse_text = ""
        max_score = 0
        best_intent = ""
        for intent in self.adapters.keys():
            intent_score = self.get_score(text, intent)
            if intent_score > max_score:
                max_score = intent_score
                best_intent = intent
        decoded_text = self.switch_case(best_intent)
        text = text.strip()
        # fill in the slots
        if "includetoken" in decoded_text:
            # run the includetoken tagger
            self.tagger_model.set_active_adapters(self.includetoken_adapter)
            tagger = TokenClassificationPipeline(model=self.tagger_model, tokenizer=self.adapter_tokenizer, task="includetoken", device=self.device)
            tagged = tagger(text)
            if len(tagged)>0:
                token_string = self.get_tagged_span(tagged, text)
                if len(token_string)==1: # in case a quote was extracted
                    decoded_text = self.check_heuristics(decoded_text, text)
                else:
                    decoded_text = decoded_text.replace("{includetoken}", token_string)
            else:
                decoded_text = self.check_heuristics(decoded_text, text)            
        if "number" in decoded_text:
            # run the number tagger
            self.tagger_model.set_active_adapters(self.number_adapter)
            tagger = TokenClassificationPipeline(model=self.tagger_model, tokenizer=self.adapter_tokenizer, task="number", device=self.device)
            tagged = tagger(text)
            if len(tagged)>0:
                token_string = self.get_tagged_span(tagged, text)
                decoded_text = decoded_text.replace("{number}", self.get_num_value(token_string))
            else:
                decoded_text = decoded_text.replace("{number}", "")
        if "id" in decoded_text:
            # run the id tagger
            self.tagger_model.set_active_adapters(self.id_adapter)
            tagger = TokenClassificationPipeline(model=self.tagger_model, tokenizer=self.adapter_tokenizer, task="id", device=self.device)
            tagged = tagger(text)
            if len(tagged)>0:
                token_string = self.get_tagged_span(tagged, text)
                decoded_text = decoded_text.replace("{id}", self.get_num_value(token_string))
            else:
                decoded_text = self.check_heuristics(decoded_text, text)
                if ("{id}" in decoded_text):
                    decoded_text = decoded_text.replace("{id}", "")
            
        app.logger.info(f"adapters decoded text {decoded_text}")      
        return None, decoded_text


    def get_score(self, text: str, intent: str):
        """Extracts the score from the intent classifier.
        """    
        self.adapter_model.set_active_adapters(self.adapters[intent])
        intent_classifier = TextClassificationPipeline(model=self.adapter_model, tokenizer=self.adapter_tokenizer, task=intent, device=self.device)
        out = intent_classifier(text)
        if len(out)>0 and out[0]['label'].replace("LABEL_","")=="1":
            score = out[0]['score']
        else:
            score = 0 
        return score
        

    def compute_grammar(self, text, error_analysis: bool = False):
        """Computes the grammar from the text.

        Arguments:
            text: the input text
            error_analysis: whether to compute extra information used for error analyses
        Returns:
            grammar: the grammar generated for the input text
            prompted_text: the prompts computed for the input text
            nn_prompts: the knn prompts, without extra information that's added for the full
                        prompted_text provided to prompt based models.
        """
        nn_prompts = None
        app.logger.info("getting prompts")
        # Compute KNN prompts
        if error_analysis:
            prompted_text, adhoc, nn_prompts = self.prompts.get_prompts(text,
                                                                        self.prompt_metric,
                                                                        self.prompt_ordering,
                                                                        error_analysis=error_analysis)
        else:
            prompted_text, adhoc = self.prompts.get_prompts(text,
                                                            self.prompt_metric,
                                                            self.prompt_ordering,
                                                            error_analysis=error_analysis)
        app.logger.info("getting grammar")
        # Compute the formal grammar, making modifications for the current input
        grammar = self.parser.get_grammar(
            adhoc_grammar_updates=adhoc)

        if error_analysis:
            return grammar, prompted_text, nn_prompts
        else:
            return grammar, prompted_text

    def update_state(self, text: str, user_session_conversation: Conversation):
        """The main conversation driver.

        The function controls state updates of the conversation. It accepts the
        user input and ultimately returns the updates to the conversation.

        Arguments:
            text: The input from the user to the conversation.
            user_session_conversation: The conversation sessions for the current user.
        Returns:
            output: The response to the user input.
        """

        if any([text is None, self.prompts is None, self.parser is None]):
            return ''

        app.logger.info(f'USER INPUT: {text}')

        # Parse user input into text abiding by formal grammar
        if self.decoding_model_name=="adapters":
            parse_tree, parsed_text = self.compute_parse_text_adapters(text)
        elif "t5" not in self.decoding_model_name:
            parse_tree, parsed_text = self.compute_parse_text(text)
        else:
            parse_tree, parsed_text = self.compute_parse_text_t5(text)

        # Run the action in the conversation corresponding to the formal grammar
        returned_item = run_action(
            user_session_conversation, parse_tree, parsed_text)

        username = user_session_conversation.username

        response_id = self.gen_almost_surely_unique_id()
        logging_info = self.build_logging_info(self.bot_name,
                                               username,
                                               response_id,
                                               text,
                                               parsed_text,
                                               returned_item)
        self.log(logging_info)
        # Concatenate final response, parse, and conversation representation
        # This is done so that we can split both the parse and final
        # response, then present all the data
        final_result = returned_item + f"<>{response_id}"

        return final_result
