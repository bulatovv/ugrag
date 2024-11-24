import polars as pl
import numpy as np
from pathlib import Path
from datasets import get_dataset_config_names
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers.errors import BulkIndexError
from elasticsearch_dsl import (
    DenseVector, Document, Text
)
    
client = Elasticsearch("http://localhost:9200", request_timeout=1200)

class Chunk(Document):
    text = Text()
    emb = DenseVector(dims=384) # Change to embedder output size
    


class DummyModel:
    def encode(self, texts, **_):
        return np.ones((len(texts), 384))

model = SentenceTransformer('all-MiniLM-L6-v2')

data_dir = Path(__file__).parent.parent.parent.resolve() / 'data'

if not data_dir.exists():
    data_dir.mkdir()

for subset in get_dataset_config_names('rungalileo/ragbench'):
    print(subset)
    q = pl.scan_parquet(f'hf://datasets/rungalileo/ragbench/{subset}')
    q = q.with_row_index('question_id')

    questions = (
        q.explode('all_relevant_sentence_keys')
        .group_by('question_id')
        .agg(
            pl.first('id', 'question'),
            pl.concat_str(
                'question_id',
                pl.lit('_'),
                'all_relevant_sentence_keys',
            ).alias('relevant_chunk_ids')
        )
    )
    
    questions_df = questions.collect()
    questions_df = questions_df.with_columns(
        embedding=model.encode(
            questions_df['question'].to_pandas(use_pyarrow_extension_array=True), # type: ignore
            show_progress_bar=True
        )
    )
    questions_df.write_parquet(data_dir / f'{subset}-documents.parquet')

    documents = (
        q.explode('documents_sentences')
        .explode('documents_sentences')
        .select(
            'question_id',
            _id=pl.concat_str(
                'question_id',
                pl.lit('_'),
                pl.col('documents_sentences').list.get(0)
            ),
            text=pl.col('documents_sentences').list.get(1)
        )

    )

    documents_df = documents.collect()
    documents_df = documents_df.with_columns(
        emb=model.encode(
            documents_df['text'].to_pandas(use_pyarrow_extension_array=True), # type: ignore
            show_progress_bar=True
        )
    )

    Chunk.init(index=subset, using=client)
    
    try:
        Chunk.bulk(
            index=subset,
            actions=(
                {
                    '_id': row['_id'], 
                    '_op_type': 'index',
                    'doc': {'text': row['text'], 'emb': row['emb']}
                }
                for row in documents_df.iter_rows(named=True)
            ),
            using=client
        )
    except BulkIndexError as e:
        for error in e.errors:
            print(error)

