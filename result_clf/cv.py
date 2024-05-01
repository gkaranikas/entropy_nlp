import json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from transformers import AutoTokenizer, DataCollatorWithPadding

from util import ner_ann_to_lines


def kfold_generator(data, k):
    fold_size = len(jdata) // k
    folds = []
    for i in range(k):
        val_indices = list(range(i*fold_size, (i+1)*fold_size))
        train_indices = list()
        for j in range(len(jdata)):
            if j not in val_indices:
                train_indices.append(j)
        val_data = [item for ix in range(len(data)) for item in data[ix] if ix in val_indices]
        train_data = [item for ix in range(len(data)) for item in data[ix] if ix in train_indices]
        dataset = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data)
        })
        yield dataset


filename = "../doccana_lines_ann.jsonl"
with open(filename, "r") as fo:
    jsonldata = list(fo)
jdata = []
for line in jsonldata:
    jdata.append(json.loads(line))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_fn(example):
    return tokenizer(example["text"], truncation=True)

#TODO add data augmentation
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

clf_metrics = evaluate.combine(['accuracy', 'precision', 'recall', 'f1'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)

evals = []
for i, dataset in enumerate(kfold_generator(data, 3)):
    tokenized_data = dataset.map(preprocess_fn)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"./classif_cv_{i}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        #eval_dataset=tokenized_data["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    eval_result = trainer.evaluate(tokenized_data['validation'])
    print(f"evals {i}:")
    print(eval_result)
    evals.append(eval_result)
    trainer.save_model()


f1 = [e["eval_f1"] for e in evals]
print(f1, sum(f1)/len(f1))
with open("./classif_cv_evals.json", "w") as fo:
    json.dump(evals, fo)