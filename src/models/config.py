# Dimensional parameters
n_paths = 8
dim_path_vectors = 26
dim_animation_types = 6
dim_animation_parameters = 6
dim_animation_vector = dim_animation_types + dim_animation_parameters

# Model one
m1_path = 'models/path_selector_extra_trees_classifier.sav'
m1_features = [f'emb_{i}' for i in range(10)] + [f'fill_{c}' for c in ['r', 'g', 'b']] + \
              ['svg_fill_r', 'diff_fill_r', 'svg_fill_g', 'diff_fill_g', 'svg_fill_b', 'diff_fill_b'] + \
              ['rel_width', 'rel_height', 'rel_x_position', 'rel_y_position', 'nr_paths_svg']

# Animation predictor
a_input_size = 26
a_hidden_sizes = [15, 20]
a_out_sizes = [dim_animation_types, dim_animation_parameters]
ap_path = 'models/ap_best_model.pth'
ap_state_dict_path = 'models/ap_best_model_state_dict.pth'

# Genetic algorithm
mutation_power = 0.2

# PCA model
pca_path = 'models/pca_path_embedding.sav'

# Scaler
scaler_path = 'models/sm_train_standard_scaler.pkl'

# Surrogate model
replacement_value = -1
s_hidden_sizes = [360, 245]
sm_fnn_path = 'models/sm_fnn.pth'
sm_tree_path = 'models/sm_gradient_boosting.sav'
sm_features = [f'emb_{i}' for i in range(10)] + \
                  ['_'.join(['fill', ch]) for ch in ['r', 'g', 'b']] + \
                  ['_'.join(['svg_fill', ch]) for ch in ['r', 'g', 'b']] + \
                  ['_'.join(['diff_fill', ch]) for ch in ['r', 'g', 'b']] + \
                  ['rel_height', 'rel_width', 'rel_x_position', 'rel_y_position',
                   'rel_x_position_to_animations', 'rel_y_position_to_animations', 'nr_paths_svg']
