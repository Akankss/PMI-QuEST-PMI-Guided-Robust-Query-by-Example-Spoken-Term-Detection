"""
pmiquest — core library for PMI-QuEST QbE-STD.

Public API
----------
from pmiquest import TFIDFBaseline, HQuEST, PMIQuest, evaluate
from pmiquest import load_corpus, load_queries, load_relevance
"""

from pmiquest.system import (  # noqa: F401
    TFIDFBaseline,
    HQuEST,
    PMIQuest,
    evaluate,
    run_comparison,
)
from pmiquest.dataloader import (  # noqa: F401
    load_corpus,
    load_queries,
    load_relevance,
)

__version__ = "1.0.0"
