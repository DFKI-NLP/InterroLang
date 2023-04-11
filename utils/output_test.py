from logic.conversation import Conversation
from logic.dataset_description import DatasetDescription
from logic.utils import read_and_format_data

TEXT = "<!DOCTYPE html><html><head></head><body>"

import gin


@gin.configurable()
def create_conversation(class_names, dataset_objective, dataset_description,
                        model_description, name,
                        dataset_file_path,
                        dataset_index_column,
                        target_variable_name,
                        categorical_features,
                        numerical_features,
                        remove_underscores):

    conversation = Conversation(class_names=class_names)
    datasetDescription = DatasetDescription(
        dataset_objective=dataset_objective,
        dataset_description=dataset_description,
        model_description=model_description, name=name)

    conversation.describe = datasetDescription

    dataset, y_values, categorical, numeric = read_and_format_data(dataset_file_path,
                                                                   dataset_index_column,
                                                                   target_variable_name,
                                                                   categorical_features,
                                                                   numerical_features,
                                                                   remove_underscores)
    conversation.add_dataset(dataset, y_values, categorical, numeric)

    conversation.build_temp_dataset()
    return conversation


gin.parse_config_file('./configs/tests.gin')

CONVERSATION = create_conversation()
