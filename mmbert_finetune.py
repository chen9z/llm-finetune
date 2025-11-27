import logging
from datasets import load_dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer
import torch

def main():
    model_name = "jhu-clsp/mmBERT-base"
    train_batch_size = 16
    num_epochs = 2
    num_hard_negatives = 7

    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="multilingual",
            license="mit",
        ),
    )
    
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(50_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=42)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=num_hard_negatives,
        margin=0,
        range_min=0,
        range_max=100,
        sampling_strategy="top",
        batch_size=2048,
        output_format="labeled-pair",
        use_faiss=True,
    )

    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )

    args = CrossEncoderTrainingArguments(
        output_dir="./mmbert-reranker",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoMSMARCO_R100_ndcg@10",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        seed=42,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=nano_beir_evaluator,
    )
    trainer.train()

    model.save_pretrained("./models/mmbert-reranker/final")

if __name__ == "__main__":
    main()
