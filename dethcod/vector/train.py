import time
from pathlib import Path
from typing import Optional

import datasets
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm.auto as tqdm
import transformers
import transformers.modeling_outputs
import typer
import wandb

from .model import VectorCompressionConfig, VectorCompressionModel

app = typer.Typer()


def train(
    dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    model: VectorCompressionModel,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    save_path: Optional[Path],
    save_interval: float = 120.0,
):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    last_save_time = time.time()
    total_loss = 0

    with tqdm.tqdm(data_loader) as pbar:
        for batch in pbar:
            input_ids = tokenizer(
                batch["text"], return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(model.device)

            model_output = model.forward(input_ids)

            num_ids = model_output.logits.shape[-1]
            logits = -F.cross_entropy(
                model_output.logits.view(-1, num_ids),
                target=input_ids.view(-1),
                ignore_index=0,
                reduction="none",
            )

            sequence_logits = logits.view(input_ids.shape).sum(dim=-1)
            acc = sequence_logits.exp().mean()
            loss = model_output.loss.item()
            total_loss += loss * batch_size

            wandb.log(
                {
                    "loss": loss,
                    "accuracy": acc,
                }
            )

            pbar.set_description(f"loss={loss:.2f}, acc={acc:.2f}")

            optimizer.zero_grad()
            model_output.loss.backward()
            optimizer.step()

            if save_path is not None and last_save_time + save_interval < time.time():
                last_save_time += save_interval
                model.save_pretrained(save_path)

        if save_path is not None:
            model.save_pretrained(save_path)


@app.command()
def main(
    model_id: str = "google-t5/t5-small",
    model_location: Path = typer.Option(
        Path("data/models/vector-t5-dethcod"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
    data_files: list[Path] = typer.Option(
        [Path("data/dataset/enwik8")],
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    lr: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model_location.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    data_files = [str(path) for path in data_files]
    dataset = datasets.load_dataset("text", data_files=data_files)
    dataset = dataset["train"]

    # TODO: try to load the model from the save location first

    t5_config = transformers.T5Config.from_pretrained(model_id)
    model_config = VectorCompressionConfig(**t5_config.to_dict())
    model = VectorCompressionModel(model_config).to(device)

    wandb.init(
        name="Vector Training",
        dir="data/wandb",
        project="DETHCOD",
        config={
            "model_id": model_id,
            "lr": lr,
            "batch_size": batch_size,
            "num_samples": len(dataset),
            "model_config": model.config.to_dict(),
        },
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    train(dataset, tokenizer, model, optimizer, batch_size, save_path=model_location)

    wandb.finish()


if __name__ == "__main__":
    app()
