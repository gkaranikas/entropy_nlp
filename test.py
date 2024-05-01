from pathlib import Path
import json
from pipeline import ClinicalNLPPipeline


def load_samples(loc):
    data = dict()
    for f in Path(loc).glob("./*.txt"):
        f_id = f.stem
        with open(f, "r") as fo:
            f_text = list(fo)
        data[f_id] = f_text
    return data


if __name__ == "__main__":
    samples = load_samples("../datasets/entropy_nlp")
    id = "0deee5f5-f9d3-4712-8d2e-c5f7cd9895dc"
    #id = "02b832e1-66cc-4f35-8b52-abf41cd821b2"
    text = samples[id]

    with open("../datasets/entropy_nlp/X1.json", "r") as fo:
        X1 = json.load(fo)

    Pipe = ClinicalNLPPipeline(X1, "./classif_results/checkpoint-500")
    output = Pipe(text)
    print(output)