import polars as pl
from pathlib import Path
from datasets import get_dataset_config_names
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
data_dir = Path(__file__).parent.parent.parent.resolve() / 'data'

if not data_dir.exists():
    data_dir.mkdir()

for subset in get_dataset_config_names('rungalileo/ragbench'):
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
            questions_df['question'].to_pandas(use_pyarrow_extension_array=True),
            show_progress_bar=True
        )
    )
    questions_df.write_parquet(data_dir / f'{subset}-documents.parquet')

    documents = (
        q.explode('documents_sentences')
        .explode('documents_sentences')
        .select(
            'question_id',
            chunk_id=pl.concat_str(
                'question_id',
                pl.lit('_'),
                pl.col('documents_sentences').list.get(0)
            ),
            chunk=pl.col('documents_sentences').list.get(1)
        )

    )

    documents_df = documents.collect()
    documents_df = documents_df.with_columns(
        embedding=model.encode(
            documents_df['chunk'].to_pandas(use_pyarrow_extension_array=True),
            show_progress_bar=True
        )
    )
    documents_df.write_parquet(data_dir / f'{subset}-questions.parquet')
