import itertools
import numpy as np
import multiprocessing
'''
Assignemnt: 
	- try to implement the branch and bound algorithm (sequentially, the in parallel)
'''

n = 10
np.random.seed(42)
C = np.random.rand(n, n)

## good point, lets make it symmetric
C = C + C.T

cntr = 0


###########################################
def evaluate(C, p):
  global cntr
  cntr = cntr + 1

  cost = 0
  for i in range(len(p)):
    cost = cost + C[p[i], p[(i + 1) % n]]
  return cost


def test_permutations(n):
  min_cost = 1000000
  for p in itertools.permutations(range(n)):
    if p[0] == 0:
      cost = evaluate(C, p)
      if cost < min_cost:
        min_cost = cost
        min_perm = p
    else:
      break

  print('>', min_perm, min_cost)


##########################################
def evaluate_bnb(C, p, min_cost):
    global cntr
    cntr += 1
    cost = sum(C[p[i]][p[i + 1]] for i in range(len(p) - 1))
    if cost < min_cost:
        return cost, p
    return min_cost, None

def test_permutations_bnb(n):
    global min_cost, min_perm
    min_cost = float('inf')
    min_perm = None
    C = [[0] * n for _ in range(n)]
    for p in itertools.permutations(range(n)):
        min_cost, perm = evaluate_bnb(C, p, min_cost)
        if perm:
            min_perm = perm
    print('>', min_perm, min_cost)


def worker(perms, C):
    local_min_cost = float('inf')
    local_min_perm = None
    for p in perms:
        cost, perm = evaluate_bnb(C, p, local_min_cost)
        if perm:
            local_min_cost = cost
            local_min_perm = perm
    return local_min_cost, local_min_perm

def parallel_test_permutations_bnb(n, num_workers=4):
    global min_cost, min_perm
    min_cost = float('inf')
    min_perm = None
    C = [[0] * n for _ in range(n)]

    perms = list(itertools.permutations(range(n)))
    chunk_size = len(perms) // num_workers
    chunks = [perms[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(worker, [(chunk, C) for chunk in chunks])
        for cost, perm in results:
            if cost < min_cost:
                min_cost = cost
                min_perm = perm

    print('>', min_perm, min_cost)

###########################################
n = 4  # Replace with actual number of cities
cntr = 0
parallel_test_permutations_bnb(n)
print('Evaluated permutations:', cntr)
