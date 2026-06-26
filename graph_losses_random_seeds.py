import matplotlib.pyplot as plt

frspecmuon = [
    0.23437438905239105,
    0.2758309245109558,
    0.23898056149482727,
    0.24100597202777863,
    0.23656393587589264,
    0.23933112621307373,
    0.24025262892246246,
    0.24430568516254425,
    0.22847795486450195,
    0.2526836693286896,
]

specmuon = [
    0.2971785068511963,
    0.2619965076446533,
    0.2788732349872589,
    0.25454995036125183,
    0.2563646137714386,
    0.25235676765441895,
    0.29717499017715454,
    0.3007761240005493,
    0.26251164078712463,
    0.2921505272388458,
]

adamw = [
    0.23540982604026794,
    0.22137078642845154,
    0.20026253163814545,
    0.19667422771453857,
    0.19817903637886047,
    0.19130896031856537,
    0.21404539048671722,
    0.2243025153875351,
    0.19908112287521362,
    0.18561948835849762,
]

runs = range(1, len(frspecmuon) + 1)

plt.figure(figsize=(8, 5))

plt.plot(runs, frspecmuon, marker='o', label='FrSpecMuon')
plt.plot(runs, specmuon, marker='s', label='SpecMuon')
plt.plot(runs, adamw, marker='^', label='AdamW')

plt.xlabel("Training Run")
plt.ylabel("Final Loss")
plt.title("Final Loss Across Training Runs")
plt.xticks(runs)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()