import gradio as gr
import torch
import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

ID2LABEL = {0: 'race', 1: 'religion', 2: 'origin', 3: 'gender', 4: 'sexuality', 5: 'age', 6: 'disability'}
LABEL2ID = {'race': 0, 'religion': 1, 'origin': 2, 'gender': 3, 'sexuality': 4, 'age': 5, 'disability': 6}

model = AutoModelForSequenceClassification.from_pretrained(
    "weights/",
    problem_type='multi_label_classification',
    num_labels=len(LABEL2ID),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

CHECKPOINT = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# setup pipeline as a text classification with multilabel outputs
hate_speech_multilabel_classifier = pipeline(
    task='text-classification',
    model=model,
    tokenizer=tokenizer,
    #device=torch.cuda.current_device(),
    top_k=None
)



def classify_hate_speech(text):
    result = hate_speech_multilabel_classifier(text)
    labels = [entry['label'] for entry in result[0]]
    scores = [entry['score'] for entry in result[0]]
    return {label: score for label, score in zip(labels, scores)}


iface = gr.Interface(
    fn=classify_hate_speech,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Label(num_top_classes=7)  # Adjust the number of top classes as needed
)

if __name__ == "__main__":
    iface.launch()
