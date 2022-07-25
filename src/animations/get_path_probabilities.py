import pickle


def get_path_probabilities(filename, animation_ids, pkl_file="data/path_selector/path_selector_label.pkl"):
    """ Function to get path probabilities that specify how likely it is that a path gets animated.

    Args:
        filename (str): SVG filename.
        animation_ids (list(int)): List of animation IDs.
        pkl_file (str): Path of pkl file which contains path labeling.

    Returns:
        list(float): List of probabilities.

    """
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)

    l = []
    for i in range(len(animation_ids)):
        try:
            probability = df[(df['filename'] == f"{filename}") & (df['animation_id'] == animation_ids[i])].iloc[0]['animate']
        except Exception as e:
            probability = 0.2
            print(f"No probability given for path with animation ID {animation_ids[i]} in logo {filename}. Probability is set to 0.2. {e}")
        l.append(probability)

    return l
