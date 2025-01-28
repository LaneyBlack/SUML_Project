# """
# This module sets up and organizes global imports for use across the project.
# It dynamically loads libraries and assigns them as global variables.
# """
#
# def setup_imports():
#     """
#     Dynamically imports and sets up commonly used libraries as global variables.
#
#     This function imports libraries like NumPy, Pandas, PyTorch, Transformers, and FastAPI,
#     and makes them globally accessible by updating the `globals()` dictionary.
#
#     Returns:
#         None
#     """
#     import numpy
#     import pandas
#     import seaborn
#     import os
#     import joblib
#     from sklearn.model_selection import train_test_split
#     from transformers import (
#     DistilBertTokenizer, DistilBertForSequenceClassification,
#     Trainer, TrainingArguments)
#     from sklearn.metrics import accuracy_score
#     from torch.utils.data import Dataset
#     import torch
#     import matplotlib.pyplot
#     from fastapi import FastAPI, HTTPException
#     import uvicorn
#     from enum import Enum
#
#     # Set globals
#     globals().update({
#         "pd": pandas,
#         "np": numpy,
#         "sns": seaborn,
#         "os": os,
#         "joblib": joblib,
#         "train_test_split": train_test_split,
#         "DistilBertTokenizer": DistilBertTokenizer,
#         "DistilBertForSequenceClassification": DistilBertForSequenceClassification,
#         "Trainer": Trainer,
#         "TrainingArguments": TrainingArguments,
#         "accuracy_score": accuracy_score,
#         "torch": torch,
#         "plt": matplotlib.pyplot,
#         "Dataset": Dataset,
#         "FastAPI": FastAPI,
#         "HTTPException": HTTPException,
#         "uvicorn": uvicorn,
#         "Enum": Enum
#     })
#
# setup_imports()