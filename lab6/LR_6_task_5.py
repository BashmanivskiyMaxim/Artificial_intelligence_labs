import numpy as np
import neurolab as nl

# Б M O
target = [
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
]

chars = ["Б", "М", "О"]
target = np.asfarray(target)
target[target == 0] = -1

# Create and train network
net = nl.net.newhop(target)

output = net.sim(target)
print("Test on train samples:")
for i in range(len(target)):
    print(chars[i], (output[i] == target[i]).all())

# Тестування для Б
print("\nTest on defaced Б:")
test = np.asfarray(
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
)
test[test == 0] = -1

out = net.sim([test])
print((out[0] == target[0]).all(), "Sim. steps", len(net.layers[0].outs))

# Тестування для М
print("\nTest on defaced М:")
test_E = np.asfarray(
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
)
test_E[test_E == 0] = -1

out_E = net.sim([test_E])
print((out_E[0] == target[1]).all(), "Sim. steps", len(net.layers[0].outs))

# Тестування для О
print("\nTest on defaced О:")
test_E = np.asfarray(
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]
)
test_E[test_E == 0] = -1

out_E = net.sim([test_E])
print((out_E[0] == target[2]).all(), "Sim. steps", len(net.layers[0].outs))
