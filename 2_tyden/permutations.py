import itertools as it

import multiprocessing as mp

D = [[0, 5, 40, 11], [5, 0, 9, 6], [40, 9, 0, 8], [11, 6, 8, 0]]

test_data = [0,5,40,11]
prefix = [0, 5]
# permutations = list(it.permutations(test_data))
def generate_permutations_with_prefix(data, prefix):
    prefix_length = len(prefix)
    remaining_data = [x for x in data if x not in prefix]
    for perm in it.permutations(remaining_data):
        yield list(prefix) + list(perm)

print(list(generate_permutations_with_prefix(test_data, prefix)))

print("--------------------")
print(list(it.permutations(test_data)))
