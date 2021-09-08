from ex4_tools import DecisionStump, decision_boundaries, generate_data
from adaboost import AdaBoost
import numpy as np
import matplotlib.pyplot as plt


###train adaboost
T=500
NUM_SAMPLES_TRAIN=5000
NUM_SAMPLES_TEST=200
NOISE_RATIO=0

X_test,y_test=generate_data(NUM_SAMPLES_TEST, NOISE_RATIO)
X_train,y_train=generate_data(NUM_SAMPLES_TRAIN, NOISE_RATIO)
boost=AdaBoost(DecisionStump, T)
boost.train(X_train,y_train)
test_err,train_err=[],[]
num_classifiers_list=np.arange(1,T)
for t in range(1,T):
    test_err.append(AdaBoost.error(boost,X_test,y_test,t))
    train_err.append(AdaBoost.error(boost,X_train, y_train, t))

plt.plot(num_classifiers_list, train_err, label='train error')
plt.plot(num_classifiers_list, test_err, label='test error')
plt.title('Adaboost error')
plt.legend()
plt.show()


for i, T in enumerate([5, 10, 50, 100, 200, 500]):
    plt.subplot(2, 3, i + 1)
    decision_boundaries(boost, X_test, y_test, T)
plt.show()

test_error = []
num_classifiers_list = np.arange(1, 500)
for T in num_classifiers_list:
    test_error.append(AdaBoost.error(boost,X_test, y_test, T))
    print(T)
T_hat = np.argmin(test_error) + 1
decision_boundaries(boost, X_train, y_train, T_hat)
plt.show()