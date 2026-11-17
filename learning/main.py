from learning.model.model import Model

model = Model()
model.learn(1000, 0.01)
model.predict(40)