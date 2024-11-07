import json
import os.path
import re
import xml.etree.ElementTree as ET
from typing import Iterator
from xml.etree import ElementTree

import evaluate
import numpy as np
import torch
from datasets import Dataset
from evaluate import EvaluationModule
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, TrainingArguments, Trainer, \
    IntervalStrategy, BatchEncoding

import pandas as pd
from transformers.utils import PaddingStrategy

print("CUDA available: ", torch.cuda.is_available())
num_labels: int
data_dir: str = os.path.abspath("aquilonia")
rpg_dir: str = os.path.join(data_dir, "RPG")
labels_path: str = os.path.join(data_dir, "handische_kategorisierung_clean.xlsx")
df: pd.DataFrame
label2int: dict[str, int] = dict()
int2label: dict[int, str] = dict()
with open(labels_path, "rb") as f:
    df = pd.read_excel(f)
    # there are 326 labels for 308 texts, i.e., ca. 18 texts have more than 1 label
    # df['sum_x'] = (df[list(df.columns)] == "x").sum(axis=1)
    delict_columns: list[str] = list(df.columns)[2:]
    for idx, label in enumerate(delict_columns):
        label2int[label] = idx
        int2label[idx] = label
    num_labels = len(delict_columns)
metric: EvaluationModule = evaluate.load("accuracy")
hf_model: str = "KoichiYasuoka/roberta-base-latin-ud-goeswith"  # "ClassCat/roberta-base-latin-v2"
tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(hf_model)
output_dir: str = os.path.join(data_dir, "test_trainer")
rpg_cache: dict[int, ElementTree] = dict()


class Cache:
    model_remote: RobertaForSequenceClassification = None
    model_local: RobertaForSequenceClassification = None

    @classmethod
    def get_model_remote(cls) -> RobertaForSequenceClassification:
        if not cls.model_remote:
            cls.model_remote = RobertaForSequenceClassification.from_pretrained(
                hf_model, num_labels=num_labels, id2label=int2label, label2id=label2int)
        return cls.model_remote

    @classmethod
    def get_model_local(cls) -> RobertaForSequenceClassification:
        if not cls.model_local:
            cls.model_local = RobertaForSequenceClassification.from_pretrained(
                output_dir, num_labels=num_labels, id2label=int2label, label2id=label2int)
        return cls.model_local


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def evaluate():
    ds: Dataset = Dataset.from_generator(get_labeled_dataset)  # get_evaluation_data
    tokenized_dataset: Dataset = ds.map(tokenize_function, batched=True).shuffle()
    trainer = Trainer(
        model=Cache.get_model_local(),
        args=TrainingArguments(output_dir=output_dir),
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics,
    )
    results: dict[str, float] = trainer.evaluate()
    print(results)


def get_evaluation_data() -> Iterator[dict]:
    with open(os.path.join(data_dir, "test.json")) as f3:
        eval_dict: dict[str, str] = json.load(f3)
        for lemma_id in eval_dict:
            # take the first column with a valid value
            label: str = eval_dict[lemma_id]
            text: str = get_rpg_text(lemma_id)
            yield {"label": label2int[label], "text": text}


def get_labeled_dataset() -> Iterator[dict]:
    idx: int
    for idx, row in df.iterrows():
        # other entries have to be retrieved online, we might implement that later
        if idx >= 259:
            break
        # take the first column with a valid value
        label: str = row.where(row == "x").dropna().keys()[0]
        lemma_id: str = str(row["Lemma-ID"])
        text: str = get_rpg_text(lemma_id)
        yield {"label": label2int[label], "text": text}


def get_rpg_text(lemma_id: str) -> str:
    rpg_volume: int = int(lemma_id[2]) + 1
    tree: ElementTree = get_rpg_volume(rpg_volume)
    root: ET.Element = tree.getroot()
    text: str = get_text_from_xml_element(root.find(f"./lemma[@id='{lemma_id}']"))
    return text


def get_rpg_volume(rpg_volume: int) -> ET.ElementTree:
    if rpg_volume not in rpg_cache:
        xml_path: str = os.path.join(rpg_dir, f"rpg{rpg_volume}.xml")
        rpg_cache[rpg_volume] = ET.parse(xml_path)
    return rpg_cache[rpg_volume]


def get_text_from_xml_element(element: ET.Element) -> str:
    xml_string: str = ElementTree.tostring(element, encoding="unicode")
    text_with_spaces: str = re.sub("<.*?>", "", xml_string)
    return ' '.join(text_with_spaces.split())


def make_predictions():
    model: RobertaForSequenceClassification = Cache.get_model_local()
    results: list[dict[str, str]] = []
    for file in tqdm([x for x in os.listdir(rpg_dir) if x.endswith(".xml")]):
        file_path: str = os.path.join(rpg_dir, file)
        tree: ElementTree = ET.parse(file_path)
        root: ET.Element = tree.getroot()
        lemmata: list[ET.Element] = root.findall("./lemma")
        lemma_ids: list[str] = [x.attrib["id"] for x in lemmata]
        texts: list[str] = [get_text_from_xml_element(x) for x in lemmata]
        batch_size: int = 200
        for i in tqdm(range(0, len(texts), batch_size)):
            texts_batch: list[str] = texts[i:i + batch_size]
            inputs: BatchEncoding = tokenizer(texts_batch, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits: torch.Tensor = model(**inputs).logits
                predicted_class_indices: torch.Tensor = np.argmax(logits, axis=1)
                predicted_classes: list[str] = [model.config.id2label[int(x)] for x in predicted_class_indices]
                for j in range(len(texts_batch)):
                    idx: int = texts.index(texts_batch[j])
                    results.append(dict(lemma_id=lemma_ids[idx], text=texts_batch[j], label=predicted_classes[j]))
    with open(os.path.join(output_dir, "predictions.json"), "w+") as f2:
        json.dump(results, f2, ensure_ascii=False)


def tokenize_function(examples):
    # TODO: remove max_length specification for other models
    return tokenizer(examples["text"], padding=PaddingStrategy.MAX_LENGTH, truncation=True, max_length=512)


def train():
    ds: Dataset = Dataset.from_generator(get_labeled_dataset)
    tokenized_dataset: Dataset = ds.map(tokenize_function, batched=True).shuffle()
    # use 10 items for development
    num_dev_items: int = 10
    small_train_dataset = tokenized_dataset.select(range(num_dev_items, len(tokenized_dataset)))
    small_eval_dataset = tokenized_dataset.select(range(num_dev_items))
    training_args: TrainingArguments = TrainingArguments(output_dir=output_dir, eval_strategy=IntervalStrategy.EPOCH,
                                                         num_train_epochs=6)
    trainer: Trainer = Trainer(
        model=Cache.get_model_remote(),
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)


# train()
# make_predictions()
evaluate()
