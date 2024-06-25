from ..vectorize_data import vectorize_data
from ..hyperparameters.user import config


s1, s2 = vectorize_data(config=config, do_prop_pred=False)

print(s1[0])
