from result_clf.pipeline import TextToResults
from rb_field_extr.pipeline import ResultsToFieldsRB
from date_extr import DateFinder
from post_proc import PostProcess


class ClinicalNLPPipeline:
    def __init__(self, lab_test_terms, model_path):
        self.terms = lab_test_terms
        self.model_path = model_path

    def __call__(self, text):
        TTRPipe = TextToResults(self.model_path)
        labels = TTRPipe(text)

        DFPipe = DateFinder()
        dates = DFPipe(text, labels)

        RTFPipe = ResultsToFieldsRB(self.terms)
        fields = RTFPipe(text, labels)

        PPPipe = PostProcess(self.terms)
        output = PPPipe(text, labels, fields, dates)

        return output
