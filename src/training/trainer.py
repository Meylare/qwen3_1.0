import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import os

# Локальные импорты (Monorepo structure)
from src.data.dataset import ViralVideoDataset
from src.models.viral_model import ViralPredictorModel

# --- КОНФИГ ---
JSONL_PATH = "data/processed/train.jsonl"
MODEL_ID = "Qwen2.5-Omni-7B"
BATCH_SIZE = 1
GRAD_ACCUMULATION_STEPS = 8
LR = 2e-4
NUM_EPOCHS = 3
CHECKPOINT_DIR = "checkpoints"


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Setup Data
    dataset = ViralVideoDataset(JSONL_PATH, processor_path=MODEL_ID, max_frames=10)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # 2. Setup Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = ViralPredictorModel(MODEL_ID, bnb_config)

    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=1.0)

    print("🚀 Starting Training...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            # Перенос данных на GPU
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            pixel_values = batch["pixel_values"].to("cuda").to(torch.float16)
            labels = batch["labels"].to("cuda").float()

            # Обработка image_grid_thw (КРИТИЧНО)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to("cuda")

            # Forward
            preds = model(input_ids, attention_mask, pixel_values, image_grid_thw)

            # Loss Calculation
            loss = criterion(preds.squeeze(), labels)
            loss = loss / GRAD_ACCUMULATION_STEPS # Normalize

            # Backward
            loss.backward()

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": loss.item() * GRAD_ACCUMULATION_STEPS})

        print(f"✅ Epoch {epoch+1} finished. Avg Loss: {total_loss / len(dataloader)}")

        # Сохранение полного состояния (LoRA + MLP Head)
        save_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Checkpoint saved: {save_path}")


if __name__ == "__main__":
    train()
