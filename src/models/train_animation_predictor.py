"""

Author's note: In a previous project state we tried using genetic algorithms to train the animation generator model.
We decided against using this training algorithm and used the aesthetics evaluation model instead
as a customized loss function. Thus, the methods in this script are to be considered deprecated.

"""

import os, sys, torch, pickle5
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from argparse import ArgumentParser

from src.models import config
from src.utils.logger import *
from src.models.genetic_algorithm import *
from src.features.get_svg_size_pos import get_relative_pos_to_bounding_box_of_animated_paths


def create_animation_vector(output, value=-1):
    """ Create the encoded animation vector from the AnimationPredictor model output.

    Args:
        output (np.array): AnimationPredictor model output.
        value (int): Value to replace predicted vector elements of other animation types with.

    Returns:
        np.array: Encoded animation vector.

    """
    for i in range(output.shape[0]):
        type_index = int(output[i][:6].argmax())
        # Set animation type vector
        for j in range(6):
            if j == type_index:
                output[i][j] = 1
            else:
                output[i][j] = 0

        # Set animation parameter vector
        if output[i][0] == 1:
            for j in [8, 9, 10, 11]:
                output[i][j] = value

        elif output[i][1] == 1:
            for j in [6, 7, 9, 10, 11]:
                output[i][j] = value

        elif output[i][2] == 1:
            for j in [6, 7, 8, 10, 11]:
                output[i][j] = value

        elif output[i][3] == 1:
            for j in [6, 7, 8, 9]:
                output[i][j] = value

        elif output[i][4] == 1:
            for j in [6, 7, 8, 9, 10, 11]:
                output[i][j] = value

        elif output[i][5] == 1:
            for j in [6, 7, 8, 9, 10, 11]:
                output[i][j] = value
    return output


def retrieve_m1_predictions(input_data):
    """ Retrieve path selector predictions to determine for each path whether to be animated or not.

    Args:
        input_data (pd.DataFrame): DataFrame containing path information.

    Returns:
        pd.DataFrame: Extended DataFrame containing additional column 'animate' which contains value 1 for animated
        and value 0 for non-animated paths.

    """
    m1_path_vectors = torch.tensor(input_data[config.m1_features].to_numpy(), dtype=torch.float)
    m1 = pickle5.load(open(config.m1_path, 'rb'))
    path_predictions = m1.predict(m1_path_vectors)
    input_data['animated'] = path_predictions

    info(f'Number of animated paths: {sum(path_predictions)}/{len(path_predictions)}')
    return input_data


def retrieve_animation_midpoints(input_data, data_dir='data/svgs_preprocessed', drop=True):
    """ Retrieve x and y position of animated path relatively to all animated paths in an SVG.

    Args:
        input_data (pd.DataFrame): DataFrame containing path information. Needs to include filename and animation_id of
        each path.
        data_dir (str): Directory containing SVG files for each logo represented in the input_data.
        drop (bool): Whether or not to drop paths that won't be animated from input_data.

    Returns:
        pd.DataFrame: Extended DataFrame containing additional columns 'rel_x_position_to_animations',
        'rel_y_position_to_animations'.

    """
    # Integrate midpoint of animation as feature
    animated_input_data = input_data[input_data['animated'] == 1]
    gb = animated_input_data.groupby('filename')['animation_id'].apply(list)

    input_data = pd.merge(left=input_data, right=gb, on='filename')
    input_data.rename(columns={'animation_id_x': 'animation_id', 'animation_id_y': 'animated_animation_ids'},
                      inplace=True)

    if drop:
        info("Non-animated paths won't be used for training")
        input_data = input_data[input_data['animated'] == 1]
    else:
        info("Non-animated paths will be used for training")
    input_data.reset_index(drop=True, inplace=True)
    info('Start extraction midpoint of animated paths as feature')
    input_data["rel_position_to_animations"] = input_data.apply(
        lambda row: get_relative_pos_to_bounding_box_of_animated_paths(f"{data_dir}/{row['filename']}.svg",
                                                                       int(row["animation_id"]),
                                                                       row["animated_animation_ids"]), axis=1)
    input_data["rel_x_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[0])
    input_data["rel_y_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[1])
    return input_data


def save_predictions(df, agents, test_paths, rewards, predictions, sorted_indices, generation):
    """ Extend dataframe containing logging information of predictions by the predictions of all agents in current
    generation and make predictions on test data.

    Args:
        df (pd.DataFrame): DataFrame containing information of previous generations.
        agents (list): List of agents of current generation.
        test_paths (np.ndarray): Test path vectors to be able to keep track of progress on test data.
        rewards (np.array): Mean average reward of each agent on training data.
        predictions (np.ndarray): Animation predictions generated by each agent on training data.
        sorted_indices (np.array): Array containing the sorted indices of agents according to their average reward.
        generation (int): Number of current generation.

    Returns:
        pd.DataFrame: Extended DataFrame containing all information of current generation.

    """
    steps = len(agents) // 5
    for i, agent in enumerate(sorted_indices):
        test_predictions = agents[agent](test_paths)
        test_reward = return_average_reward(test_paths, test_predictions)

        train_types = list()
        for prediction in predictions[agent]:
            train_types.append(np.argmax(prediction.detach().numpy()))
        train_type_counts = Counter(train_types)

        test_types = list()
        for prediction in test_predictions:
            test_types.append(np.argmax(prediction.detach().numpy()))
        test_type_counts = Counter(test_types)

        df = df.append(
            {'generation': generation, 'agent': agent, 'agent_rank': i,
             'train_mean_reward': rewards[agent], 'test_mean_reward': test_reward,
             'train_translate': train_type_counts[0], 'train_scale': train_type_counts[1],
             'train_rotate': train_type_counts[2], 'train_skew': train_type_counts[3],
             'train_fill': train_type_counts[4], 'train_opacity': train_type_counts[5],
             'test_translate': test_type_counts[0], 'test_scale': test_type_counts[1],
             'test_rotate': test_type_counts[2], 'test_skew': test_type_counts[3],
             'test_fill': test_type_counts[4], 'test_opacity': test_type_counts[5]},
            ignore_index=True)
    return df


def get_n_types(predictions):
    """ Get number of animation type predictions.

    Args:
        predictions (np.ndarray): Animation predictions to retrieve number of animation types from.

    Returns:
        dict: Number of predictions for each animation type.

    """
    n_types = list()
    for agent_prediction in predictions:
        types = list()
        for prediction in agent_prediction:
            types.append(np.argmax(prediction.detach().numpy()))
        counts = Counter(types)
        n_types.append(len(counts))
    return n_types


def train_animation_predictor(train_paths, test_paths, hidden_sizes=config.a_hidden_sizes, out_sizes=config.a_out_sizes,
                              num_agents=100, top_parent_limit=10, generations=10, timestamp=''):
    """ Iterate over a given number of generations and train animation predictor model using genetic algorithm.

    Args:
        train_paths (torch.Tensor): Path vectors of training dataset.
        test_paths (torch.Tensor): Path vectors of test dataset.
        hidden_sizes (list): Number of neurons in each hidden layer. Must be of length=2.
        out_sizes (list): Number of neurons in each output layer. Must be of length=2.
        num_agents (int): Number of agents to be considered in each generation.
        top_parent_limit (int): Number of top parents to be considered for each subsequent generation.
        generations (int): Number of generation to train for.
        timestamp (str): Timestamp of starting time of training (used for logging).

    Returns:
        src.models.animation_prediction.AnimationPredictor: Best agent after training.

    """
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize number of agents
    overall_start = datetime.now()
    agents = create_random_agents(num_agents=num_agents)

    info('AP model summary')
    print('=' * 100)
    print(f'Number of training instances: {len(train_paths)}')
    print(f'Number of agents: {num_agents}')
    print(f'Top parent limit: {top_parent_limit}')
    print(f'Number of generations: {generations}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Output size: {out_sizes}')
    print('=' * 100)

    training_process = pd.DataFrame(
        {'generation': [], 'agent': [], 'agent_rank': [],
         'train_mean_reward': [], 'test_mean_reward': [],
         'train_translate': [], 'train_scale': [],
         'train_rotate': [], 'train_skew': [],
         'train_fill': [], 'train_opacity': [],
         'test_translate': [], 'test_scale': [],
         'test_rotate': [], 'test_skew': [],
         'test_fill': [], 'test_opacity': []})

    for generation in range(generations):
        start = datetime.now()
        info(f'Generation {generation + 1}/{generations}')
        rewards, predictions = compute_agent_rewards(agents=agents, path_vectors=train_paths)
        sorted_parent_indexes = np.argsort(rewards)[::-1].astype(int)
        training_process = save_predictions(df=training_process, agents=agents, test_paths=test_paths,
                                            rewards=rewards, predictions=predictions,
                                            sorted_indices=sorted_parent_indexes, generation=generation)
        sorted_parent_indexes = sorted_parent_indexes[:top_parent_limit]
        top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        children_agents = crossover(agents=top_agents, num_agents=num_agents)
        children_agents = [mutate(agent) for agent in children_agents]

        agents = children_agents

        stop = datetime.now()
        info(f'Operation time: {stop - start}')

        print('-' * 100)

        print(f'Mean rewards: {np.mean(rewards)} | Mean of top 10: {np.mean(top_rewards[:10])}')
        print(f'Top {top_parent_limit} agents: {sorted_parent_indexes}')
        print(f'Rewards for top {top_parent_limit} agents: {top_rewards[:top_parent_limit]}')

        print('=' * 100)

    overall_stop = datetime.now()
    info(f'Overall operation time: {overall_stop - overall_start}')

    if not os.path.exists('logs'):
        os.makedirs('logs')
    training_process.to_csv(f'logs/{timestamp}_animation_predictions.csv', index=False)

    return top_agents[0]


def main(train_path='data/path_selector/path_selector_train.csv', test_path='data/path_selector/path_selector_test.csv', drop=True,
         num_agents=100, top_parent_limit=20, generations=50, timestamp='', model1=True):
    """ Main function to prepare and run training using the genetic algorithm.

    Args:
        train_path (str): Path to training data.
        test_path (str): Path to test data.
        drop (bool): Whether or not to drop paths that won't be animated from input_data.
        num_agents (int): Number of agents to be considered in each generation.
        top_parent_limit (int): Number of top parents to be considered for each subsequent generation.
        generations (int): Number of generation to train for.
        timestamp (str): Timestamp of starting time of training (used for logging).
        model1 (bool): Whether or not to apply path selector model and scaling before training.

    Returns:
        src.models.animation_prediction.AnimationPredictor: Best agent after training.

    """
    info(f'Train data source: {train_path}')
    info(f'Test data source: {test_path}')

    info(f'Surrogate model in use: {config.sm_fnn_path}')
    info(f'Replacement value for animation vector elements not in use: {config.replacement_value}')

    info(f'Mutation power: {config.mutation_power}')

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if model1:
        # Apply model one to get predictions whether to animate paths or not
        train_data = retrieve_m1_predictions(train_data)
        test_data = retrieve_m1_predictions(test_data)

        # Retrieve features describing the midpoint of animated paths
        train_data = retrieve_animation_midpoints(train_data, drop=drop)
        test_data = retrieve_animation_midpoints(test_data, drop=drop)

        # Scale input data for surrogate model
        scaler = pickle5.load(open(config.scaler_path, 'rb'))
        train_data[config.sm_features] = scaler.transform(train_data[config.sm_features])
        test_data[config.sm_features] = scaler.transform(test_data[config.sm_features])
        info('Scaled input data for surrogate model')
    else:
        info("Model 1 and scaling won't be applied to input data")

    # Prepare path vectors for animation prediction
    train_paths = torch.tensor(train_data[config.sm_features].to_numpy(), dtype=torch.float)
    test_paths = torch.tensor(test_data[config.sm_features].to_numpy(), dtype=torch.float)

    # Perform genetic algorithm for animation prediction
    top_model = train_animation_predictor(train_paths=train_paths, test_paths=test_paths, num_agents=num_agents,
                                          top_parent_limit=top_parent_limit, generations=generations,
                                          timestamp=timestamp)

    # Save best model for animation prediction
    # torch.save(top_model, f'models/{timestamp}_ap_best_model.pkl')
    # torch.save(top_model.state_dict(), f'models/{timestamp}_ap_best_model_state_dict.pth')

    return top_model


if __name__ == '__main__':
    os.chdir('../..')

    ap = ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-s", "--save", required=False, action='store_true', help="if set, output will be saved to file")
    ap.add_argument("-d", "--drop", required=False, action='store_false', help="if set, non-animated paths "
                                                                               "will be kept for training")
    ap.add_argument("-m1", "--model1", required=False, action='store_false', help="if set, model one won't be applied "
                                                                                  "to input data. Note: Also expects "
                                                                                  "input data to be scaled already")
    ap.add_argument("-train", "--train", required=False, default='data/path_selector'
                                                                 '/path_selector_train.csv',
                    help="path to training data")
    ap.add_argument("-test", "--test", required=False, default='data/path_selector/path_selector_test.csv',
                    help="path to test data")

    ap.add_argument("-a", "--n_agents", required=False, default=100, help="number of agents")
    ap.add_argument("-t", "--top_parents", required=False, default=20, help="number of top agents to be considered, "
                                                                            "should be even number")
    ap.add_argument("-g", "--generations", required=False, default=50, help="number of generations")
    args = vars(ap.parse_args())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args['save']:
        Path('logs/').mkdir(parents=True, exist_ok=True)
        log_file = f"logs/{timestamp}_ap_training.txt"
        info(f'Saving enabled, log file: {log_file}')
        sys.stdout = open(log_file, 'w')
        main(train_path=args['train'], test_path=args['test'],
             num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp, model1=args['model1'])
        sys.stdout.close()
    else:
        main(train_path=args['train'], test_path=args['test'],
             num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp, model1=args['model1'])
