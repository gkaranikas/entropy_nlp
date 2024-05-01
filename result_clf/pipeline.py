from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class TextToResults:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="cpu",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline('text-classification',
                             model=self.model,
                             tokenizer=self.tokenizer)

    def __call__(self, lines):
        return self.pipe(lines)
