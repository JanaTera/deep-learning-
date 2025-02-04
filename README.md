 #Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Create the XOR Inputs and Outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Build the Model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))  
# Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the Model
model.fit(X, y, epochs=1000, verbose=1)
# Plot the decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid)
    predictions = (predictions > 0.5).astype(int)
    plt.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', marker='o', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary for XOR Problem')
    plt.show()

plot_decision_boundary(X, y, model)
# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f'Accuracy: {accuracy * 100:.2f}%')
