import os
from collections import Counter

import conllu
import pandas
import pandas as pd
from conllu import SentenceList

pos_dir: str = os.path.abspath("pos_data")
counters: dict[str, Counter] = dict()
for file in os.listdir(pos_dir):
    if not file.endswith("conllu"):
        continue
    with open(os.path.join(pos_dir, file)) as f:
        sl: SentenceList = conllu.parse(f.read())
        counter: Counter = Counter([y["upostag"] for x in sl for y in x])
        # print(file)
        # keys: list[str] = list(counter.keys())
        # print(",".join(keys))
        # print(",".join([str(counter[x]) for x in keys]))
        counters[file] = counter
all_pos: set[str] = set([y for x in counters.values() for y in x.keys()])
df: pd.DataFrame = pd.DataFrame(columns=list(all_pos))
for file, counter in counters.items():
    for key in counter.keys():
        df.at[file, key] = counter[key]
df.to_excel(os.path.join(pos_dir, "pos_data.xlsx"))
