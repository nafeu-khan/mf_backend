import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier


def hyperparameter_optimization(X_train, y_train):
	do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")
	if do_hyperparameter_optimization:
		st.subheader("Hyperparameter Optimization Settings")
		n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=5, step=1)
		cv = st.number_input("Number of cross-validation folds", min_value=2, value=2, step=1)
		random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

	if "perceptron_best_param" not in st.session_state:
		st.session_state.perceptron_best_param = {
			"hidden_layer_sizes": 3,
			"activation": "relu",
			"solver": "adam",
			"alpha": 0.0001,
			"learning_rate_init": 0.001,
			"max_iter": 1000,
			"tol": 0.001
		}

	st.write('#')

	if do_hyperparameter_optimization and st.button('Run Optimization'):

		st.write('#')
		param_dist = {
			'hidden_layer_sizes': [3,10, 50, 100],
			'activation': ['identity', 'logistic', 'tanh', 'relu'],
			'solver': ['lbfgs', 'sgd', 'adam'],
			'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
			'learning_rate_init': [0.001, 0.01, 0.1, 1],
			'max_iter': [100, 200, 500, 1000],
			'tol': [0.0001, 0.001, 0.01]
		}

		model = MLPClassifier()

		with st.spinner('Doing hyperparameter optimization...'):

			clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
									 random_state=random_state)
			st.spinner("Fitting the model...")
			clf.fit(X_train, y_train)

		st.success("Hyperparameter optimization completed!")

		cv_results = clf.cv_results_

		param_names = list(cv_results['params'][0].keys())

		# Create a list of dictionaries with the parameter values and accuracy score for each iteration
		results_list = []
		for i in range(len(cv_results['params'])):
			param_dict = {}
			for param in param_names:
				param_dict[param] = cv_results['params'][i][param]
			param_dict['accuracy'] = cv_results['mean_test_score'][i]
			results_list.append(param_dict)

		results_df = pd.DataFrame(results_list)

		results_df = results_df.sort_values(by=['accuracy'], ascending=False)

		if "opti_results_df" not in st.session_state:
			st.session_state.opti_results_df = pd.DataFrame

		st.session_state.opti_results_df = results_df

		best_param = clf.best_params_

		st.session_state.perceptron_best_param = best_param

	return st.session_state.perceptron_best_param


def perceptron(X_train, y_train):

	best_param = hyperparameter_optimization(X_train, y_train)

	try:
		st.write('#')
		st.write(st.session_state.opti_results_df)
	except:
		pass

	st.subheader("Model Settings")

	col1, col2, col3 = st.columns(3)
	with col1:
		hidden_size = st.number_input(
			"Hidden Layer Size",
			min_value=1, max_value=200, step=1,
			key="perceptron_hidden_size",
			value=best_param.get("hidden_layer_sizes", 1)
		)

		alpha = st.number_input(
			"Alpha",
			min_value=1e-6, max_value=100.0, step=1e-4,
			key="perceptron_alpha",
			format="%f",
			value=best_param.get("alpha", 1e-4)
		)

	with col2:
		activation = st.selectbox(
			"Activation Function",
			["identity", "logistic", "tanh", "relu"],
			index=["identity", "logistic", "tanh", "relu"].index(best_param.get("activation", "relu")),
			key="perceptron_activation",
		)

		learning_rate = st.number_input(
			"Learning Rate",
			min_value=1e-6, max_value=1.0, step=1e-3,
			key="perceptron_lr",
			format="%f",
			value=best_param.get("learning_rate_init", 1e-3)
		)

	with col3:
		max_iter = st.number_input(
			"Max Iteration",
			min_value=1, max_value=1000000, step=1,
			key="perceptron_max_iter",
			value=best_param.get("max_iter", 200)
		)

		tol = st.number_input(
			"Tolerance (ε)",
			min_value=1e-8, max_value=1.0, step=1e-4,
			format="%f",
			key="perceptron_tol",
			value=best_param.get("tol", 1e-4)
		)

	cols = st.columns(int(hidden_size))
	hidden_layer_sizes = []
	for i in range(int(hidden_size)):
		neuron_size = best_param.get("hidden_layer_sizes", 100)
		# st.number_input(
		# 	f"Layer {i + 1} Neuron Size",
		# 	min_value=1, max_value=10000, step=1,
		# 	key=f"percenptron_neuron_size_{i}",
		# 	value=best_param.get("hidden_layer_sizes", 100)
		# )
		hidden_layer_sizes.append(neuron_size)


	# Display table of hidden layer sizes
	table_data = {"Layer": [f"Layer {i + 1}" for i in range(len(hidden_layer_sizes))],
				  "Neuron Size": hidden_layer_sizes}
	table_data=st.experimental_data_editor(table_data)


	try:
		model = MLPClassifier(hidden_layer_sizes=table_data["Neuron Size"], activation=activation, alpha=alpha,
							  learning_rate_init=learning_rate, max_iter=max_iter, tol=tol)
	except ValueError as ve:
		print(f"ValueError: {ve}")
	except TypeError as te:
		print(f"TypeError: {te}")
	except Exception as e:
		print(f"Error: {e}")

	return model