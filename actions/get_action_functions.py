"""Contains a function returning a dict mapping the key word for actions to the function.

This function is used to generate a dictionary of all the actions and the corresponding function.
This functionality is used later on to determine the set of allowable operations and what functions
to run when parsing the grammar.
"""
from actions.about.define import define_operation
from actions.about.function import function_operation
from actions.about.model import model_operation
from actions.about.self import self_operation
from actions.context.followup import followup_operation
from actions.context.last_turn_filter import last_turn_filter
from actions.context.last_turn_operation import last_turn_operation
#from actions.explanation.counterfactuals import counterfactuals_operation
from actions.explanation.feature_importance import feature_importance_operation
from actions.explanation.rationalize import rationalize_operation
#from actions.explanation.topk import global_topk_operation
#from actions.explanation.what_if important what_if_operation
from actions.filter.filter import filter_operation
from actions.filter.includes_token import includes_operation
from actions.global_topk import global_top_k
from actions.metadata.count_data_points import count_data_points
from actions.metadata.data_summary import data_operation
from actions.metadata.feature_stats import feature_stats
from actions.metadata.labels import show_labels_operation
from actions.metadata.show_data import show_operation
from actions.nlu.sentiment import sentiment_operation
from actions.nlu.similarity import similar_instances_operation
from actions.nlu.topic import topic_operation
from actions.prediction.mistakes import show_mistakes_operation
from actions.prediction.predict import predict_operation
from actions.prediction.prediction_likelihood import predict_likelihood
from actions.prediction.score import score_operation


def get_all_action_functions_map():
    """Gets a dictionary mapping all the names of the actions in the parse tree to their functions."""
    actions = {
        'countdata': count_data_points,
        'filter': filter_operation,
        'predict': predict_operation,
        'self': self_operation,
        'previousfilter': last_turn_filter,
        'previousoperation': last_turn_operation,
        'data': data_operation,
        'followup': followup_operation,
        'show': show_operation,
        #'change': what_if_operation,
        'likelihood': predict_likelihood,
        'model': model_operation,
        'function': function_operation,
        'score': score_operation,
        'label': show_labels_operation,
        'mistake': show_mistakes_operation,
        'statistic': feature_stats,
        'define': define_operation,
        'predictionfilter': filter_operation,
        'labelfilter': filter_operation,
        'includes': includes_operation,
        'similarity': similar_instances_operation,
        'topic': topic_operation,
        'sentiment': sentiment_operation,
        'rationalize': rationalize_operation,
        'feature_importance': feature_importance_operation,
        'globaltopk': global_top_k
    }
    return actions
