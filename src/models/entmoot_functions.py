from entmoot.optimizer.optimizer import Optimizer

import copy
import inspect
import numbers
import pickle
import time
import pandas as pd
from src.models.blackbox_sm_fnn import *
from entmoot.space.space import Space
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def entmoot_fit(dimensions, x0, y0, base_estimator="GBRT", std_estimator="BDD", random_state=None):
    """ Fits ENTMOOT optimizer as described in
    Thebelt, Kronqvist, Mistry, Lee, Sudermann-Merx, and Misener (2020)
    *ENTMOOT: A Framework for Optimization over Ensemble Tree Models*
    https://arxiv.org/abs/2003.04774.

    Args:
        dimensions (list):  List of search space dimensions.
                            Each search dimension can be defined either as
                            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
                              dimensions),
                            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
                              dimensions),
                            - as a list of categories (for `Categorical` dimensions), or
                            - an instance of a `Dimension` object (`Real`, `Integer` or
                              `Categorical`)
        x0 (list): Initial input points.
                   - If it is a list of lists, use it as a list of input points.
                   - If it is a list, use it as a single initial input point.
                   - If it is `None`, no initial input points are used.
        y0 (list): Evaluation of initial input points.
                   - If it is a list, then it corresponds to evaluations of the function
                     at each element of `x0` : the i-th element of `y0` corresponds
                     to the function evaluated at the i-th element of `x0`.
                   - If it is a scalar, then it corresponds to the evaluation of the
                     function at `x0`.
                   - If it is None and `x0` is provided, then the function is evaluated
                     at each element of `x0`.
        base_estimator (str): A default LightGBM surrogate model of the corresponding type is used
                              to fit the initial data.
                              The following model types are available:
                              - "GBRT" for gradient-boosted trees
                              - "RF" for random forests
        std_estimator (str): A model is used to estimate uncertainty of `base_estimator`.
                             Different types can be classified as exploration measures, i.e. move
                             as far as possible from reference points, and penalty measures, i.e.
                             stay as close as possible to reference points. Within these types, the
                             following uncertainty estimators are available:
                             - exploration:
                                - "BDD" for bounded-data distance, which uses squared euclidean
                                  distance to standardized data points
                                - "L1BDD" for bounded-data distance, which uses manhattan
                                  distance to standardized data point
                             - penalty:
                                - "DDP" for data distance, which uses squared euclidean
                                  distance to standardized data points
                                - "L1DDP" for data distance, which uses manhattan
                                  distance to standardized data points
        random_state (int): Set random state to something other than None for reproducible results.

    Returns:
        object: Fitted Optimizer object that can be used to predict new inputs.

    """

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if not x0:
        raise ValueError("Please provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    # calculate the total number of initial points
    n_initial_points = len(x0)

    # Build optimizer
    # create optimizer class
    opt = Optimizer(
        dimensions,
        base_estimator=base_estimator,
        std_estimator=std_estimator,
        n_initial_points=n_initial_points,
        initial_point_generator="random",
        acq_func="LCB",
        acq_optimizer="global",
        random_state=random_state,
        acq_func_kwargs=None,
        acq_optimizer_kwargs={},
        base_estimator_kwargs=None,
        std_estimator_kwargs=None,
        model_queue_size=None,
        verbose=0
    )

    # record initial points through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        opt.tell(x0, y0)

    print('-'*60)
    print("Fitted a model to observed evaluations of the objective.")

    return opt


def entmoot_predict(opt, func, path_vector, n_calls):
    """ Predicts optimial animation based on given optimizer and path vector.

    Args:
        opt (object): Fitted Optimizer object, obtained by `entmoot_fit`.
        func (object): BenchmarkFunction object that is used for funtion evaluations, e.g. SurrogateModelFNN().
        path_vector(list): List that defines path vector (26-dim).

    Returns:
        list, int: Optimal vector based on the given path vector / optimizer and corresponding score.

    """
    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    # initialize the search space manually
    space = Space(func.get_bounds())

    # get the core of the gurobi model from helper function 'get_core_gurobi_model'
    core_model = get_core_gurobi_model(space)

    # define variables
    an_vec_0 = core_model._cont_var_dict[0]
    an_vec_1 = core_model._cont_var_dict[1]
    an_vec_2 = core_model._cont_var_dict[2]
    an_vec_3 = core_model._cont_var_dict[3]
    an_vec_4 = core_model._cont_var_dict[4]
    an_vec_5 = core_model._cont_var_dict[5]
    an_vec_6 = core_model._cont_var_dict[6]
    an_vec_7 = core_model._cont_var_dict[7]
    an_vec_8 = core_model._cont_var_dict[8]
    an_vec_9 = core_model._cont_var_dict[9]
    an_vec_10 = core_model._cont_var_dict[10]
    an_vec_11 = core_model._cont_var_dict[11]
    emb_0 = core_model._cont_var_dict[12]
    emb_1 = core_model._cont_var_dict[13]
    emb_2 = core_model._cont_var_dict[14]
    emb_3 = core_model._cont_var_dict[15]
    emb_4 = core_model._cont_var_dict[16]
    emb_5 = core_model._cont_var_dict[17]
    emb_6 = core_model._cont_var_dict[18]
    emb_7 = core_model._cont_var_dict[19]
    emb_8 = core_model._cont_var_dict[20]
    emb_9 = core_model._cont_var_dict[21]
    fill_r = core_model._cont_var_dict[22]
    fill_g = core_model._cont_var_dict[23]
    fill_b = core_model._cont_var_dict[24]
    svg_fill_r = core_model._cont_var_dict[25]
    svg_fill_g = core_model._cont_var_dict[26]
    svg_fill_b = core_model._cont_var_dict[27]
    diff_fill_r = core_model._cont_var_dict[28]
    diff_fill_g = core_model._cont_var_dict[29]
    diff_fill_b = core_model._cont_var_dict[30]
    rel_height = core_model._cont_var_dict[31]
    rel_width = core_model._cont_var_dict[32]
    rel_x_position = core_model._cont_var_dict[33]
    rel_y_position = core_model._cont_var_dict[34]
    rel_x_position_to_animations = core_model._cont_var_dict[35]
    rel_y_position_to_animations = core_model._cont_var_dict[36]
    nr_paths_svg = core_model._cont_var_dict[37]

    # add constraints to define animation vector structure
    core_model.addConstr(an_vec_0 + an_vec_1 + an_vec_2 + an_vec_3 + an_vec_4 + an_vec_5 == 1)

    core_model.addGenConstrIndicator(an_vec_0, True, an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)

    core_model.addGenConstrIndicator(an_vec_1, True, an_vec_6 + an_vec_7 + an_vec_9 + an_vec_10 + an_vec_11 == 0)

    core_model.addGenConstrIndicator(an_vec_2, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_10 + an_vec_11 == 0)

    core_model.addGenConstrIndicator(an_vec_3, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 == 0)

    core_model.addGenConstrIndicator(an_vec_4, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)

    core_model.addGenConstrIndicator(an_vec_5, True, an_vec_6 + an_vec_7 + an_vec_8 + an_vec_9 + an_vec_10 + an_vec_11 == 0)

    # add constraints to keep path vector fixed
    core_model.addConstr(emb_0 == path_vector[0])
    core_model.addConstr(emb_1 == path_vector[1])
    core_model.addConstr(emb_2 == path_vector[2])
    core_model.addConstr(emb_3 == path_vector[3])
    core_model.addConstr(emb_4 == path_vector[4])
    core_model.addConstr(emb_5 == path_vector[5])
    core_model.addConstr(emb_6 == path_vector[6])
    core_model.addConstr(emb_7 == path_vector[7])
    core_model.addConstr(emb_8 == path_vector[8])
    core_model.addConstr(emb_9 == path_vector[9])
    core_model.addConstr(fill_r == path_vector[10])
    core_model.addConstr(fill_g == path_vector[11])
    core_model.addConstr(fill_b == path_vector[12])
    core_model.addConstr(svg_fill_r == path_vector[13])
    core_model.addConstr(svg_fill_g == path_vector[14])
    core_model.addConstr(svg_fill_b == path_vector[15])
    core_model.addConstr(diff_fill_r == path_vector[16])
    core_model.addConstr(diff_fill_g == path_vector[17])
    core_model.addConstr(diff_fill_b == path_vector[18])
    core_model.addConstr(rel_height == path_vector[19])
    core_model.addConstr(rel_width == path_vector[20])
    core_model.addConstr(rel_x_position == path_vector[21])
    core_model.addConstr(rel_y_position == path_vector[22])
    core_model.addConstr(rel_x_position_to_animations == path_vector[23])
    core_model.addConstr(rel_y_position_to_animations == path_vector[24])
    core_model.addConstr(nr_paths_svg == path_vector[25])

    core_model.update()

    opt.acq_optimizer_kwargs['add_model_core'] = core_model
    opt.update_next()

    result = None

    _n_calls = n_calls

    while _n_calls > 0:
        # predict animation vector based on fitted Optimizer model and evaluate corresponding score using given surrogate model
        _n_calls -= 1

        next_x = opt.ask()
        next_y = func(next_x)

        #if itr == 1:
        #    best_fun = min(next_y)

        #itr += 1

        result = opt.tell(next_x, next_y, fit=_n_calls>0)

        #best_fun = result.fun
        result.specs = specs

    x_iters = result['x_iters'][-n_calls:]
    func_vals = result['func_vals'][-n_calls:]
    min_i = np.argmin(func_vals)
    #return next_x, next_y

    return x_iters[min_i], func_vals[min_i], result


if __name__ == "__main__":
    initial_data = pd.read_csv('../../data/entmoot_optimization/bo_initial_data_09042021.csv')

    X_train = initial_data.iloc[:100, :-4]
    X_train.replace(to_replace=-1, value=0, inplace=True
                    )
    y_train = initial_data.iloc[:100, -4:]

    y_train = pd.Series(decode_classes(y_train.to_numpy()).flatten()) * -1

    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()

    # Load surrogate model
    func = SurrogateModelFNN()

    opt = entmoot_fit(dimensions=func.get_bounds(), x0=X_train, y0=y_train, base_estimator="RF", std_estimator="MPI", random_state=73)

#    with open("../../models/entmoot_optimizer_100_old.pkl", "rb") as f:
#        opt = pickle.load(f)

    path_vector = [-0.5301350924026049,
 -0.5169283152132866,
 -0.7181804353056276,
 -0.5852317288579015,
 -0.0439171920917359,
 -0.109643896491656,
 -0.9075397748406688,
 -0.8081042288431435,
 0.6236006066727882,
 -0.2824519451971694,
 -1.4745267257932293,
 -0.3366244069648179,
 -0.8107911925048927,
 -0.8076373072108983,
 0.4425432126236162,
 -1.1309958047879736,
 -0.964979971370144,
 -0.7333761386913004,
 0.0111370522991624,
 -0.728046421028966,
 -0.8089747448340847,
 -0.3860035457854465,
 -1.0842525983849884,
 0.3856467552605218,
 -0.9240113273333622,
 1.0388136084635364]

    best_x, best_y, _ = entmoot_predict(opt, func, path_vector, n_calls=5)
    print(best_x)
    print(best_y)