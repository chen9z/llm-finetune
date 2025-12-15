import logging
from pprint import pprint
from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name=""
    eval_batch_size =64

    model=CrossEncoder(model_name)
    hard_eval_dataset = load_dataset("tomaarsen/gooaq-reranker-blogpost-datasets", "rerank", split="eval")

    samples = [
        {
            "query": sample["question"],
            "positive": [sample["answer"]],
            "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
        }
        for sample in hard_eval_dataset
    ]
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=samples,
        batch_size=eval_batch_size,
        name="gooaq-dev-realistic",
        always_rerank_positives=False,
    )
    realistic_results = reranking_evaluator(model)
    pprint(realistic_results)

    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=samples,
        batch_size=eval_batch_size,
        name="gooaq-dev-evaluation",
        always_rerank_positives=True,
    )
    evaluation_results = reranking_evaluator(model)
    pprint(evaluation_results)

if __name__=="__main__":
    main()

