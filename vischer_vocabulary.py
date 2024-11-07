import csv
import json
import os.path
import subprocess

from tqdm import tqdm

latmor_analyze_path: str = "echo {0} | ./LatMor/fst-infl LatMor/latmor.a"
latmor_generate_path: str = "echo '{0}' | ./LatMor/fst-infl LatMor/latmor-gen.a"
vischer_output_path: str = "vocabulary_vischer.json"


def convert_infinitives_to_1sg(lemmata: list[str]):
    inf_string: str = "<inf>"
    ind_string: str = "<ind>"
    new_lemmata: list[str] = []
    with open(os.devnull, "w") as dev_null:
        for lemma in tqdm(lemmata):
            result_bytes: bytes = subprocess.check_output(latmor_analyze_path.format(lemma), shell=True,
                                                          stderr=dev_null)
            result_string: str = result_bytes.decode('utf-8')
            analyses: list[str] = result_string.split("\n")[1:-1]
            new_lemma: str = lemma
            for analysis in analyses:
                if inf_string in analysis:
                    first_sg: str = analysis.replace(inf_string, ind_string if ind_string not in analysis else "")
                    first_sg += "<sg><1>"
                    result_bytes = subprocess.check_output(latmor_generate_path.format(first_sg), shell=True,
                                                           stderr=dev_null)
                    result_string = result_bytes.decode('utf-8')
                    first_sg_generated: str = result_string.split("\n")[1]
                    new_lemma = first_sg_generated
                    break
            new_lemmata.append(new_lemma)
    json.dump(new_lemmata, open(vischer_output_path, "w+"))


def adjust_vischer_vocabulary():
    vischer_csv_path: str = os.path.abspath("Vischer.csv")
    # ignore first 3 lines
    rows: list[list[str]] = list(csv.reader(open(vischer_csv_path)))[3:]
    # remove second column, it is empty anyway; remove trailing whitespace
    rows_single_column: list[str] = [x[0].strip() for x in rows]
    # remove anything after the first comma (usually inflected forms)
    rows_no_comma: list[str] = [x.split(",")[0] for x in rows_single_column]
    # remove anything after the first slash (usually inflected forms)
    rows_no_slash: list[str] = [x.split("/")[0] for x in rows_no_comma]
    replacements: dict[str, str] = {" Stf": "", " =?": "", ".": "", "(": "", ")": "", "ú": "u", "…": ""}
    rows_with_replacements: list[str] = rows_no_slash
    # replace/remove strange remarks
    for replacement in replacements:
        rows_with_replacements = [x.replace(replacement, replacements[replacement]) for x in rows_with_replacements]
    # remove anything after first whitespace (usually collocations or references)
    rows_no_whitespace: list[str] = [x.split()[0] for x in rows_with_replacements]
    # remove duplicate entries
    rows_deduplicated: list[str] = sorted(list(set(rows_no_whitespace)))
    convert_infinitives_to_1sg(rows_deduplicated)


def remove_macra():
    entries: list[str] = json.load(open(vischer_output_path))
    replacements: dict[str, str] = {"ō": "o", "ū": "u", "ē": "e", "ī": "i", "ā": "a"}
    for i in tqdm(range(len(entries))):
        for key in replacements:
            entries[i] = entries[i].replace(key, replacements[key])
    json.dump(entries, open(vischer_output_path, "w+"))


entries: list[str] = json.load(open(vischer_output_path))
entries_deduplicated: list[str] = sorted(list(set(entries)))
json.dump(entries_deduplicated, open(vischer_output_path, "w+"))
a = 0
