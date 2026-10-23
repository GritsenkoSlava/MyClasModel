class Update:
    def __init__(self):
        pass

    def update(self, old_w, old_b, grad_w, grad_b, learning_rate):
        new_w = old_w - learning_rate * grad_w
        new_b = old_b - learning_rate * grad_b

        return new_w, new_b