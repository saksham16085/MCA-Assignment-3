import numpy as np
from scipy import sparse as sp
import sys
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim.copy()
    vec_docs = vec_docs.toarray()
    vec_queries = vec_queries.toarray()
    alpha = 0.68
    beta = 0.32
    n_epochs = 3
    for i in range(n_epochs):
        for j in range(0, vec_queries.shape[0]):
            top = np.argsort(-rf_sim[:, j])[:n]
            for k in top:
                vec_queries[j] += vec_docs[k]*alpha
            bottom = np.argsort(rf_sim[:, j])[:n]
            for k in bottom:
                vec_queries[j] -= vec_docs[k]*beta
        rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim.copy()
    alpha = 0.74
    beta = 0.26
    number_of_terms = 8
    vec_docs = vec_docs.toarray()
    vec_queries = vec_queries.toarray()
    np.set_printoptions(threshold=sys.maxsize)
    inv_matrix = tfidf_model.inverse_transform(vec_docs)

    inv_dic = {v: k for k, v in tfidf_model.vocabulary_.items()}
    dic = {k: v for k, v in tfidf_model.vocabulary_.items()}
    A = np.zeros((9493, 1033))
    for i in range(1033):
        doc = inv_matrix[i]
        for j in range(len(doc)):
            key = dic[doc[j]]
            A[key][i] += 1
    # print(A.shape)
    c = np.matmul(A, A.transpose())

    for i in range(3):
        for j in range(0, vec_queries.shape[0]):
            new_term_set = []
            top = np.argsort(-rf_sim[:, j])[:n]
            # Get the k most relevant documents.
            for k in top:
                vec_queries[j] += vec_docs[k] * alpha
            bottom = np.argsort(rf_sim[:, j])[:n]
            # Get the k most irrelevant documents
            for k in bottom:
                vec_queries[j] -= vec_docs[k] * beta
            top_term_indices = np.argsort(-vec_queries[j, :])[:number_of_terms]
            for o in top_term_indices:
                tp = np.argsort(-c[o, :])[:number_of_terms]
                words = []
                for k in range(len(tp)):
                    words.append(inv_dic[tp[k]])
                delta = tfidf_model.transform([' '.join(words)]).toarray()[0]
                vec_queries[j] += delta

        rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim
