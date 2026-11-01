from model import Model

model = Model()
model.learn(10000, 0.01)
print()
model.test(60)