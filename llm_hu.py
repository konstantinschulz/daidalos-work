import csv
import json

from gradio_client import Client


def annotate():
    # uses https://huggingface.co/CohereForAI/c4ai-command-r-plus internally
    client = Client("https://llm1-compute.cms.hu-berlin.de/")  # llm3
    with open("lemmata_ancient_greek_proiel.json") as f:
        words: list[str] = json.load(f)
        words = words[8950:]  # 2450 already done
        max_word_count: int = 1000
        chunks: list[list[str]] = [words]
        # input list is too large, need to split into chunks
        if len(words) > max_word_count:
            chunks = [words[i: i + max_word_count] for i in range(0, len(words), max_word_count)]
        for chunk in chunks:
            result = client.predict(
                f"In the following, I will provide a list of Ancient Greek words to you. Add sentiment labels to each word according to the following schema: -1 (very negative), -0.5 (slightly negative), 0 (neutral), 0.5 (slightly positive), 1 (very positive). Use this output format 'GREEK_WORD: NUMERIC_LABEL', putting each entry on a new line, without index numbers, copying the GREEK_WORD exactly from the input. Do not add explanations or notes to the output. Here comes the list: \n ``` {chunk} ```",
                # str in 'parameter_1' Textbox component
                api_name="/chat"
            )
            print(result)


tsv: csv.reader = csv.reader(open("sentiment_lexicon_ancient_greek_cmd-r-plus_unordered.tsv"), delimiter="\t")
sentiment_lexicon: dict[str, float] = dict()
for row in tsv:
    sentiment_lexicon[row[0]] = float(row[1])
sentiment_lexicon_ordered: list[tuple[str, float]] = list(sorted(sentiment_lexicon.items(), key=lambda x: x[0]))
lexicon_string: str = "\n".join([f"{x[0]}\t{x[1]}" for x in sentiment_lexicon_ordered])
lexicon_string = lexicon_string.replace(".0", "")
with open("sentiment_lexicon_ancient_greek_cmd-r-plus.tsv", "w+") as f:
    f.write(lexicon_string)
