# import matplotlib.pyplot as plt


def get_policies():
    """return a list of tuples (index, path, reward_mean)"""
    with open("./pongdeterministic-v4/performance_log.txt", 'r') as f:
        lines = f.readlines()
        summary_lines = []
        for l in lines:
            if l[0:5] == "index":
                summary_lines.append(l)

    all_policies = []
    for l in summary_lines:
        temp_l = l.split(',')
        m = (int(temp_l[0].strip()[6:]), temp_l[1].strip()[5:], float(temp_l[2].strip()[13:]))
        all_policies.append(m)

    all_policies.sort(key=lambda tup: tup[2])
    # rewards = [r[2] for r in all_policies]
    #
    # plt.plot(range(len(rewards)), rewards, 'o')
    # plt.xlabel("index of the policy")
    # plt.ylabel("average reward of 200 tests")
    # plt.show()

    index_20 = [1, 7, 11,  23, 35, 41, 45, 58, 61, 63, 67, 69, 71, 75, 76, 78, 82, 87, 97, 107]
    policies = [all_policies[i] for i in index_20]

    return policies

if __name__ == "__main__":
    policies = get_policies()
    for i, p in enumerate(policies):
        print(i, p)