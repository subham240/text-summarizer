from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

model.to(device)
model.eval()


# --------------------------------------------------
# Templates
# --------------------------------------------------

templates = Jinja2Templates(directory=".")


# --------------------------------------------------
# Input Model
# --------------------------------------------------

class DialogueInput(BaseModel):
    dialogue: str


# --------------------------------------------------
# Clean Data
# --------------------------------------------------

def clean_data(text: str):

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# --------------------------------------------------
# Summarization
# --------------------------------------------------

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# --------------------------------------------------
# Summarize API
# --------------------------------------------------

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )