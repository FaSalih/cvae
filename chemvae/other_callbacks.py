# import numpy as np
# from keras.api.callbacks import Callback

# class RmseCallback(Callback):
#     """
#     This is executed in this fashion:

#     model.fit(np.array([[1.0]]), np.array([[1.0]]),
#            callbacks=[MyCallback()])

#     Is there a way to get `self.model` type ?

#     This fn isn't actually used
#     """
#     def __init__(self, X_test, Y_test, parameters, df_norm: pd.DataFrame|None = None):
#         super().__init__()
#         self.df_norm = df_norm
#         self.X_test = X_test
#         self.Y_test = Y_test
#         self.config = parameters

#     def on_epoch_end(self, epoch, logs=None):
#         df_norm = self.df_norm
#         X_test = self.X_test
#         Y_test = self.Y_test

#         if isinstance(self.model, Model):
#             y_pred = self.model.predict(X_test,self.config['batch_size'])
#         if type(y_pred) is list:
#             if 'reg_prop_tasks' in self.config and 'logit_prop_tasks' in self.config:
#                 y_pred = y_pred[-2]
#             elif 'reg_prop_tasks' in self.config:
#                 y_pred = y_pred[-1]
#         if df_norm is not None:
#             std = df_norm['std'].values
#             mean =df_norm['mean'].values
#             # unsure how to especify the type
#             y_pred = y_pred *  std + mean #un norm the predictions
#             Y_test = Y_test * std + mean # and the tests.

#         rmse = np.sqrt(np.mean(np.square(y_pred - Y_test), axis=0))
#         mae = np.mean(np.abs(y_pred - Y_test), axis=0)
#         if df_norm is not None:
#             df_norm['rmse'] = rmse
#             df_norm['mae'] = mae
#             print("RMSE test set:", df_norm['rmse'].to_dict())
#             print("MAE test set:", df_norm['mae'].to_dict())
#         else:
#             if "reg_prop_tasks" in self.config:
#                 print("RMSE test set:", self.config["reg_prop_tasks"], rmse)
#                 print("MAE test set:", self.config["reg_prop_tasks"], mae)
#             else:
#                 print("RMSE test set:", rmse)
#                 print("MAE test set:", mae)

# def sample(a, temperature=0.01):
#     """Sample from the list of probabilities (derived from `a`).

#     This function does not seem to be used anywhere.
#     a: list of logits.
#     Think of `a` the probability of each outcome, for dice with len(a) faces
#     [p_0, p_1, p_2,...]
#     temperature: a modulator.

#     In practice this will likely be used to select the dice's face (mol) with that statistics.
#     """
#     a = np.log(a) / temperature
#     a = np.exp(a) / np.sum(np.exp(a))  # probabilities
#     return np.argmax(np.random.multinomial(1, a, 1))  # rolls the dice 1 time.
