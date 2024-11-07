import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D

# Fixing random state for reproducibility
np.random.seed(19680801)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
erat: list[list[int]] = [[2, 3, 4], [4, 3, 6], [7, 3, 4]]
verbum: list[list[int]] = [[3, 4, 5], [5, 4, 3], [3, 4]]
deus: list[list[int]] = [[6, 7, 5], [5, 7, 3]]
labels: list[str] = ["erat", "verbum", "deus"]
coords: list[list[float]] = []
for word_idx, word in enumerate([erat, verbum, deus]):
    avg_vector: list[float] = []
    for idx in range(3):
        values: list[int] = [x[idx] for x in word if len(x) > idx]
        avg: float = float(np.average(values))
        avg_vector.append(avg)
    coords.append(avg_vector)
    ax.scatter(avg_vector[0], avg_vector[1], avg_vector[2], label=labels[word_idx])

print("erat vs. verbum: ", np.linalg.norm(np.array(coords[0]) - np.array(coords[1])))
# scipy.spatial.distance.cosine(coords[0], coords[1]) scipy.spatial.distance.cosine(coords[1], coords[2])
print("verbum vs. deus", np.linalg.norm(np.array(coords[1]) - np.array(coords[2])))
lines: list[Line2D] = ax.plot([x[0] for x in coords], [x[1] for x in coords], [x[2] for x in coords], c="lightgray")
ax.annotate("Abstand: 3.52", (.5, -.1), xycoords=lines[0], ha='center', va='bottom')
ax.annotate("Abstand: 1.37", (0.02, .1), xycoords=lines[0], ha='center', va='bottom', rotation=75)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title("3-dimensionaler Vektorraum mit Euklidischem Abstand")
ax.legend()
fig.savefig('embeddings_workshop_3d.png', dpi=600)
plt.show()
