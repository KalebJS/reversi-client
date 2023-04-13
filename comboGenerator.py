import json

results = []
for i in range(101):
    for j in range(101 - i):
        for k in range(101 - i - j):
            for l in range(101 - i - j - k):
                m = 100 - i - j - k - l
                results.append([i, j, k, l, m])

# generate all possible sets
combinations = []
for i in range(32768):
    binary = bin(i)[2:].zfill(15)
    combination = {}
    subset_sums = []
    for j in range(5):
        start = j * 3
        subset = binary[start:start+3]
        subset_sum = int(subset, 2)
        subset_sums.append(subset_sum)
    total_sum = sum(subset_sums)
    # adjust subset sums to add up to 100
    adjusted_sums = [int(subset_sum * 100 / total_sum) if total_sum != 0 else 0 for subset_sum in subset_sums]
    diff = 100 - sum(adjusted_sums)
    # adjust the last subset sum to make the total sum 100
    adjusted_sums[-1] += diff
    combination["set1"] = [adjusted_sums[0]] * 8
    combination["set2"] = [adjusted_sums[1]] * 8
    combination["set3"] = [adjusted_sums[2]] * 8
    combination["set4"] = [adjusted_sums[3]] * 8
    combination["set5"] = [adjusted_sums[4]] * 8
    combination["group_num"] = i
    combination["attempted"] = False
    combination["soft_wins"] = 0
    combination["easy_wins"] = 0
    combination["medium_wins"] = 0
    combination["hard_wins"] = 0
    combination["usable"] = False
    combinations.append(combination)

# write results to file
with open('combinations.json', 'w') as f:
    json.dump(combinations, f)