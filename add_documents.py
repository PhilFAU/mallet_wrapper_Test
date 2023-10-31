def add_documents(self, documents, prune_at=2000000):
    """Update dictionary from a collection of `documents`.

    Parameters
    ----------
    documents : iterable of iterable of str
        Input corpus. All tokens should be already **tokenized and normalized**.
    prune_at : int, optional
        Dictionary will try to keep no more than `prune_at` words in its mapping, to limit its RAM
        footprint, the correctness is not guaranteed.
        Use :meth:`~gensim.corpora.dictionary.Dictionary.filter_extremes` to perform proper filtering.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from mallet_wrapper.corpora import Dictionary
        >>>
        >>> corpus = ["máma mele maso".split(), "ema má máma".split()]
        >>> dct = Dictionary(corpus)
        >>> len(dct)
        5
        >>> dct.add_documents([["this", "is", "sparta"], ["just", "joking"]])
        >>> len(dct)
        10

    """
    for docno, document in enumerate(documents):
        # log progress & run a regular check for pruning, once every 10k docs
        if docno % 10000 == 0:
            if prune_at is not None and len(self) > prune_at:
                self.filter_extremes(no_below=0, no_above=1.0, keep_n=prune_at)
            logger.info("adding document #%i to %s", docno, self)

        # update Dictionary with the document
        self.doc2bow(document, allow_update=True)  # ignore the result, here we only care about updating token ids

    logger.info(
        "built %s from %i documents (total %i corpus positions)",
        self, self.num_docs, self.num_pos
    )