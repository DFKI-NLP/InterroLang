from logic.conversation import Conversation
from logic.dataset_description import DatasetDescription

TEXT = "<!DOCTYPE html><html><head></head><body>"


def create_conversation():
    conversation = Conversation(class_names={1: "True", 0: "False"})

    datasetDescription = DatasetDescription(
        dataset_objective="predict to answer yes/no questions based on text passages",
        dataset_description="Boolean question answering (yes/no)",
        model_description="DistilBERT", name="boolq")

    conversation.describe = datasetDescription

    return conversation
