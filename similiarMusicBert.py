import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import miditoolkit
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

def log_print(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

class MusicBERTEmbedding(nn.Module):
    def __init__(self, vocab_size=6000):
        super().__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=0,
            output_hidden_states=True,
            output_attentions=False
        )
        self.encoder = BertModel(config)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_4 = torch.stack(output.hidden_states[-4:])
        cls_vectors = last_4[:, :, 0, :].mean(dim=0)
        return cls_vectors

def midi_to_remi_tokens(filepath, max_length=512, device='cpu'):
    midi_obj = miditoolkit.midi.parser.MidiFile(filepath)
    events = []
    for track in midi_obj.instruments:
        if track.is_drum: continue
        for note in track.notes:
            start = min(int(note.start // 10), 999)
            duration = min(int((note.end - note.start) // 10), 999)
            pitch = note.pitch
            velocity = min(note.velocity, 127)
            events.append((start, duration, pitch, velocity))
    events.sort()
    tokens = []
    for start, duration, pitch, velocity in events:
        tokens += [1000+start, 2000+duration, 3000+pitch, 4000+velocity]
    tokens = tokens[:max_length] + [0]*(max_length-len(tokens))
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    mask = (input_ids != 0).long()
    return input_ids, mask

def collect_midi_files(root_dir, log_file):
    midi_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mid'):
                midi_files.append(os.path.join(dirpath, filename))
    log_print(f"📂 {root_dir} 에서 {len(midi_files)}개 MIDI 파일 발견", log_file)
    return midi_files

class MIDISimilarityDataset(Dataset):
    def __init__(self, file_list, device):
        self.file_list = file_list
        self.device = device

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file1 = self.file_list[idx]
        same_dir = os.path.dirname(file1)
        pos_files = [f for f in os.listdir(same_dir) if f.endswith('.mid') and os.path.join(same_dir, f) != file1]
        if pos_files:
            file2 = os.path.join(same_dir, random.choice(pos_files))
            label = 1
        else:
            file2 = random.choice(self.file_list)
            while os.path.dirname(file2) == same_dir:
                file2 = random.choice(self.file_list)
            label = -1
        input_ids_1, mask_1 = midi_to_remi_tokens(file1, device=self.device)
        input_ids_2, mask_2 = midi_to_remi_tokens(file2, device=self.device)
        target = torch.tensor(label, dtype=torch.float, device=self.device)
        return input_ids_1, mask_1, input_ids_2, mask_2, target

def compute_accuracy(vec1, vec2, targets, threshold=0.5):
    cosine_sim = F.cosine_similarity(vec1, vec2)
    preds = torch.where(cosine_sim >= threshold, torch.tensor(1.0, device=vec1.device), torch.tensor(-1.0, device=vec1.device))
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total

def train(model, dataloader, optimizer, loss_fn, log_file):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    log_print(f"🚀 Training 시작: {len(dataloader)} batch", log_file)
    for input_ids_1, mask_1, input_ids_2, mask_2, target in tqdm(dataloader, desc="Training"):
        vec1 = model(input_ids_1.squeeze(1), mask_1.squeeze(1))
        vec2 = model(input_ids_2.squeeze(1), mask_2.squeeze(1))
        loss = loss_fn(vec1, vec2, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        correct, total = compute_accuracy(vec1, vec2, target)
        total_correct += correct
        total_samples += total
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    log_print(f"📈 Training Accuracy: {avg_acc:.4f}", log_file)
    return avg_loss, avg_acc

def validate(model, dataloader, loss_fn, log_file):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    log_print(f"🔍 Validation 시작: {len(dataloader)} batch", log_file)
    with torch.no_grad():
        for input_ids_1, mask_1, input_ids_2, mask_2, target in tqdm(dataloader, desc="Validation"):
            vec1 = model(input_ids_1.squeeze(1), mask_1.squeeze(1))
            vec2 = model(input_ids_2.squeeze(1), mask_2.squeeze(1))
            loss = loss_fn(vec1, vec2, target)
            total_loss += loss.item()
            correct, total = compute_accuracy(vec1, vec2, target)
            total_correct += correct
            total_samples += total
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    log_print(f"📈 Validation Accuracy: {avg_acc:.4f}", log_file)
    return avg_loss, avg_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_file = "training_log.txt"
model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"📅 Training Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

model = MusicBERTEmbedding(vocab_size=5000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CosineEmbeddingLoss()

train_files = collect_midi_files(r'D:\ListenToMyHeartBeat\140.음악 유사성 판별 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL', log_file)
val_files = collect_midi_files(r'D:\ListenToMyHeartBeat\140.음악 유사성 판별 데이터\01-1.정식개방데이터\Validation\02.라벨링데이터\VL', log_file)

train_dataset = MIDISimilarityDataset(train_files, device=device)
val_dataset = MIDISimilarityDataset(val_files, device=device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

epochs = 5
for epoch in range(epochs):
    log_print(f"\n🌀 Epoch {epoch+1}/{epochs} 시작: {datetime.now().strftime('%H:%M:%S')}", log_file)
    train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, log_file)
    val_loss, val_acc = validate(model, val_loader, loss_fn, log_file)
    log_print(f" Epoch {epoch+1} 완료: {datetime.now().strftime('%H:%M:%S')}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", log_file)
    
    model_save_path = os.path.join(model_save_dir, f"model_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), model_save_path)

# 🌟 최종 모델 저장
final_model_path = os.path.join(model_save_dir, "model_final.pt")
torch.save(model.state_dict(), final_model_path)
log_print(f"\n🏁 Training 완료 및 최종 모델 저장: {final_model_path}", log_file)
