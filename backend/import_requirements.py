def setup_imports():
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from tqdm import tqdm
    from scipy.sparse import vstack

    # Set globals
    globals().update({
        "pd": pd,
        "np": np,
        "joblib": joblib,
        "train_test_split": train_test_split,
        "TfidfVectorizer": TfidfVectorizer,
        "LinearSVC": LinearSVC,
        "tqdm": tqdm,
        "vstack": vstack,
    })


setup_imports()

pd = globals().get("pd")
np = globals().get("np")
joblib = globals().get("joblib")
train_test_split = globals().get("train_test_split")
sns = globals().get("sns")
TfidfVectorizer = globals().get("TfidfVectorizer")
tqdm = globals().get("tqdm")
vstack = globals().get("vstack")
LinearSVC = globals().get("LinearSVC")
