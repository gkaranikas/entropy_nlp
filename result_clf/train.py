import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from transformers import AutoTokenizer, DataCollatorWithPadding

def ner_ann_to_lines(jdata):

    def overlaps(a, b, c, d):
        if b <= c:
            return -1
        if d <= a:
            return 1
        else:
            return 0

    txt = jdata["text"]
    lbl = jdata["label"]
    lbl_idx = 0
    lines = []
    i = 0
    i_start = 0
    while i < len(txt):
        if i == len(txt)-1 or txt[i] == '\n':
            i = i+1
            while lbl_idx < len(lbl) and (ov:=overlaps(lbl[lbl_idx][0], lbl[lbl_idx][1], i_start, i)) == -1:
                lbl_idx += 1
            if ov == 0:
                lines.append({"text":txt[i_start:i], "label":1})
            else:
                lines.append({"text":txt[i_start:i], "label":0})
            i_start = i
        else:
            i = i+1
    return lines


filename = "../doccana_lines_ann.jsonl"
with open(filename, "r") as fo:
    jsonldata = list(fo)
jdata = []
for line in jsonldata:
    jdata.append(json.loads(line))

dataset = Dataset.from_list([item for jd in jdata for item in ner_ann_to_lines(jd)])
dataset = dataset.train_test_split(test_size=0.3)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_fn(example):
    return tokenizer(example["text"], truncation=True)

tokenized_data = dataset.map(preprocess_fn)
#TODO add data augmentation
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./classif_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)

clf_metrics = evaluate.combine(['accuracy', 'precision', 'recall', 'f1'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    #eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

trainer.evaluate(tokenized_data['test'])

trainer.save_model()