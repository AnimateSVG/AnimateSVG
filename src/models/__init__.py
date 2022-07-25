"""
.. autoclass:: src.models.animation_prediction.AnimationPredictor
    :members:
.. autoclass:: src.models.blackbox_sm_fnn.SurrogateModelFNN
    :members:
.. automodule:: src.models.coral_loss
    :members:
.. automodule:: src.models.genetic_algorithm
    :members:
.. autoclass:: src.models.ordinal_classifier_fnn.OrdinalClassifierFNN
    :members:
.. automodule:: src.models.train_animation_predictor
    :members:
"""
from .animation_prediction import AnimationPredictor
from .blackbox_sm_fnn import SurrogateModelFNN
from .coral_loss import coral_loss
from .entmoot_functions import entmoot_fit, entmoot_predict
from .genetic_algorithm import init_weights, create_random_agents, create_animation_vector, prepare_sm_input, \
    return_average_reward, compute_agent_rewards, crossover, mutate
from .ordinal_classifier_fnn import OrdinalClassifierFNN, predict
from .ordinal_classifier_scikit import OrdinalClassifier, RandomForestOC, GradientBoostingOC, ExtraTreesOC
from .train_animation_predictor import retrieve_m1_predictions, retrieve_animation_midpoints, \
    save_predictions, get_n_types
