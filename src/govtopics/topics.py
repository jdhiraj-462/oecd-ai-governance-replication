"""Topic models: BERTopic (primary) and LDA (robustness check)."""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import adjusted_rand_score


def stopwords(cfg: dict) -> list[str]:
    return sorted(set(ENGLISH_STOP_WORDS) | set(cfg["text"]["extra_stopwords"]))


# ---------------------------------------------------------------- embeddings
def embed(texts: list[str], cfg: dict, backend: str = "sbert") -> np.ndarray:
    """'sbert' is the analysis setting. 'tfidf-svd' exists only so the test suite
    runs offline; it is never used for reported results."""
    if backend == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["bertopic"]["embedding_model"])
        return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    if backend == "tfidf-svd":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(stop_words=stopwords(cfg), min_df=2).fit_transform(texts)
        k = max(2, min(100, X.shape[1] - 1, X.shape[0] - 1))
        Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
        return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)
    raise ValueError(backend)


# ---------------------------------------------------------------- BERTopic
def fit_bertopic(texts: list[str], emb: np.ndarray, cfg: dict, seed: int):
    import hdbscan
    import umap
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer

    b, t = cfg["bertopic"], cfg["text"]
    um = umap.UMAP(random_state=seed, **b["umap"])
    hd = hdbscan.HDBSCAN(prediction_data=True, **b["hdbscan"])
    vec = CountVectorizer(stop_words=stopwords(cfg), ngram_range=tuple(t["ngram_range"]),
                          min_df=b["vectorizer"]["min_df"], max_df=b["vectorizer"]["max_df"])
    model = BERTopic(umap_model=um, hdbscan_model=hd, vectorizer_model=vec,
                     ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(texts, embeddings=emb)
    n_outliers_raw = int(np.sum(np.array(topics) == -1))
    if b["reduce_outliers"] and n_outliers_raw and len(set(topics)) > 1:
        topics = model.reduce_outliers(texts, topics, strategy="c-tf-idf")
        model.update_topics(texts, topics=topics, vectorizer_model=vec,
                            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True))
    return model, np.array(topics), n_outliers_raw


def bertopic_stability(texts, emb, cfg) -> pd.DataFrame:
    """Pairwise adjusted Rand index between topic assignments from different seeds."""
    assign = {s: fit_bertopic(texts, emb, cfg, s)[1] for s in cfg["bertopic"]["seeds"]}
    rows = [{"seed_a": a, "seed_b": b, "ari": adjusted_rand_score(assign[a], assign[b]),
             "k_a": len(set(assign[a]) - {-1}), "k_b": len(set(assign[b]) - {-1})}
            for a, b in combinations(assign, 2)]
    return pd.DataFrame(rows)


def topic_table(model, topics: np.ndarray, n_words: int = 10) -> pd.DataFrame:
    rows = []
    for tid in sorted(set(topics)):
        words = [w for w, _ in (model.get_topic(tid) or [])][:n_words]
        rows.append({"topic": int(tid), "n_docs": int((topics == tid).sum()),
                     "share": float((topics == tid).mean()), "top_terms": ", ".join(words),
                     "label": ""})  # 'label' is filled by the authors in topic_labels.csv
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- coherence
def tokenize(texts: list[str], cfg: dict) -> list[list[str]]:
    sw = set(stopwords(cfg))
    toks = [[w for w in re.findall(r"[a-z][a-z\-]{2,}", t.lower()) if w not in sw] for t in texts]
    from gensim.models.phrases import Phraser, Phrases
    bigram = Phraser(Phrases(toks, min_count=5, threshold=10))
    return [bigram[d] for d in toks]


def coherence(topic_words: list[list[str]], tokens: list[list[str]], measure: str) -> float:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    d = Dictionary(tokens)
    tw = [[w for w in ws if w in d.token2id] for ws in topic_words]
    tw = [ws for ws in tw if len(ws) >= 3]
    if not tw:
        return float("nan")
    return float(CoherenceModel(topics=tw, texts=tokens, dictionary=d,
                                coherence=measure, topn=10).get_coherence())


def bertopic_words_as_tokens(model, topics, n=10):
    out = []
    for tid in sorted(set(topics) - {-1}):
        out.append([w.replace(" ", "_") for w, _ in (model.get_topic(tid) or [])][:n])
    return out


# ---------------------------------------------------------------- LDA
def fit_lda(tokens: list[list[str]], k: int, cfg: dict, seed: int):
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    d = Dictionary(tokens)
    n = len(tokens)
    d.filter_extremes(no_below=cfg["text"]["min_df"], no_above=cfg["text"]["max_df"])
    bow = [d.doc2bow(t) for t in tokens]
    L = cfg["lda"]
    m = LdaModel(bow, id2word=d, num_topics=k, passes=L["passes"], iterations=L["iterations"],
                 alpha=L["alpha"], eta=L["eta"], random_state=seed, eval_every=None)
    theta = np.zeros((n, k))
    for i, b in enumerate(bow):
        for j, p in m.get_document_topics(b, minimum_probability=0.0):
            theta[i, j] = p
    return m, theta


def lda_grid(tokens, cfg) -> pd.DataFrame:
    rows = []
    for k in cfg["lda"]["k_grid"]:
        for s in cfg["lda"]["seeds"]:
            m, _ = fit_lda(tokens, k, cfg, s)
            words = [[w for w, _ in m.show_topic(j, 10)] for j in range(k)]
            rows.append({"k": k, "seed": s, "c_v": coherence(words, tokens, "c_v"),
                         "c_npmi": coherence(words, tokens, "c_npmi")})
    return pd.DataFrame(rows)
