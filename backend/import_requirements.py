def setup_imports():
    import numpy
    import pandas
    import seaborn
    import os
    import joblib
    from sklearn.model_selection import train_test_split
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import accuracy_score
    from torch.utils.data import Dataset
    import torch
    import matplotlib.pyplot
    from fastapi import FastAPI, HTTPException
    import uvicorn
    from enum import Enum

    # Set globals
    globals().update({
        "pd": pandas,
        "np": numpy,
        "sns": seaborn,
        "os": os,
        "joblib": joblib,
        "train_test_split": train_test_split,
        "DistilBertTokenizer": DistilBertTokenizer,
        "DistilBertForSequenceClassification": DistilBertForSequenceClassification,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "accuracy_score": accuracy_score,
        "torch": torch,
        "plt": matplotlib.pyplot,
        "Dataset": Dataset,
        "FastAPI": FastAPI,
        "HTTPException": HTTPException,
        "uvicorn": uvicorn,
        "Enum": Enum
    })


setup_imports()

pd = globals().get("pd")
np = globals().get("np")
os = globals().get("os")
joblib = globals().get("joblib")
train_test_split = globals().get("train_test_split")
sns = globals().get("sns")

DistilBertTokenizer = globals().get("DistilBertTokenizer")
DistilBertForSequenceClassification = globals().get("DistilBertForSequenceClassification")
Trainer = globals().get("Trainer")
TrainingArguments = globals().get("TrainingArguments")
accuracy_score = globals().get("accuracy_score")

torch = globals().get("torch")
plt = globals().get("plt")
Dataset = globals().get("Dataset")

FastAPI = globals().get("FastAPI")
HTTPException = globals().get("HTTPException")
uvicorn = globals().get("uvicorn")
Enum = globals().get("Enum")
