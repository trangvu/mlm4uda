import matplotlib.pyplot as plt

TASK_NAMES = [ "wnut2016", "fin", "jnlpba", "bionlp09" ]
# TASK_NAMES = ["wnut2016"]
STRATEGIES = [ "rand", "pos", "entropy", "adv" ]

DIR = "/home/trang/workspace/repo/advLM/adv/experiments/analysis/density_ratio"
task_ratio = {}
task_rank = {}
for task in TASK_NAMES:
    ratio = {}
    rank = {}
    for strategy in STRATEGIES:
        r = []
        with open("{}/ratio_{}_{}.txt".format(DIR,task, strategy)) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    r.append(float(line))
        ratio[strategy] = r

        rk = []
        with open("{}/rank_{}_{}.txt".format(DIR,task, strategy)) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rk.append(float(line))
        rank[strategy] = rk
    task_ratio[task] = ratio
    task_rank[task] = rank

labels = [2500, 5000, 7500, 10000, 12500, 15000, 17500 ,20000 ,22500, 25000]
markers = ["+", "v", "s", "."]
colors = [ "purple", "green", "blue","red"]

fig, ax = plt.subplots(nrows=2, ncols=2)
index = -1

for row in ax:
    for col in row:
        index += 1
        task_name = TASK_NAMES[index]
        ratio = task_ratio[task_name]
        col.plot(labels, ratio['rand'], marker=markers[0], color=colors[0], label="random")
        col.plot(labels, ratio['pos'], marker=markers[1], color=colors[1], label="pos")
        col.plot(labels, ratio['entropy'], marker=markers[2], color=colors[2], label="entropy")
        col.plot(labels, ratio['adv'], marker=markers[3], color=colors[3], label="adv")
        # col.xlabel('Steps')
        # col.ylabel('average r(w)')
        col.set(title=task_name)
ax[0][1].legend()
# plt.show()
fig.tight_layout(pad=1.0)
plt.savefig('{}/lexical_ratio.png'.format(DIR, task))

plt.clf()
fig, ax = plt.subplots(nrows=2, ncols=2)
index = -1

for row in ax:
    for col in row:
        index += 1
        task_name = TASK_NAMES[index]
        rank = task_rank[task_name]
        col.plot(labels, rank['rand'], marker=markers[0], color=colors[0], label="random")
        col.plot(labels, rank['pos'], marker=markers[1], color=colors[1], label="pos")
        col.plot(labels, rank['entropy'], marker=markers[2], color=colors[2], label="entropy")
        col.plot(labels, rank['adv'], marker=markers[3], color=colors[3], label="adv")
        # col.xlabel('Steps')
        # col.ylabel('average r(w)')
        col.set(title=task_name)
ax[0][1].legend()
# plt.show()
fig.tight_layout(pad=1.0)
plt.savefig('{}/lexical_rank.png'.format(DIR, task))