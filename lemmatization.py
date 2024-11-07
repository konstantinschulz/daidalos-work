import json
import os

import conllu
from conllu import SentenceList

tb_dir: str = os.path.abspath("UD_Ancient_Greek-PROIEL")
sls: SentenceList = SentenceList()
for file in os.listdir(tb_dir):
    if not file.endswith(".conllu"):
        continue
    path: str = os.path.join(tb_dir, file)
    with open(path) as f:
        sl: SentenceList = conllu.parse(f.read())
        sls += sl
lemmata: set[str] = set([token["lemma"] for token_list in sls for token in token_list])
with open("lemmata_ancient_greek_proiel.json", "w+") as f2:
    json.dump(sorted(lemmata), f2, ensure_ascii=False, indent=2)
