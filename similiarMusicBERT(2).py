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
import csv

# 로그 출력되게
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
        parts1 = file1.split(os.sep)
        song_id1 = parts1[-3]
        type1 = parts1[-2]

        # 유사(1): 같은 곡번호 & 하나는 Org, 하나는 Cover
        candidates_pos = [f for f in self.file_list if f != file1 and 
                          f.split(os.sep)[-3] == song_id1 and 
                          ((type1 == 'Org' and f.split(os.sep)[-2] != 'Org') or 
                           (type1 != 'Org' and f.split(os.sep)[-2] == 'Org'))]
        if candidates_pos:
            file2 = random.choice(candidates_pos)
            label = 1
        else:
            # 비유사(-1): 같은 곡번호 내 Cover-Cover 또는 다른 곡번호
            candidates_neg = [f for f in self.file_list if f != file1 and 
                              (f.split(os.sep)[-3] != song_id1 or 
                               (f.split(os.sep)[-3] == song_id1 and f.split(os.sep)[-2] != 'Org' and type1 != 'Org'))]
            if candidates_neg:
                file2 = random.choice(candidates_neg)
                label = -1
            else:
                # fallback: 자기 자신
                file2 = file1
                label = 1

        input_ids_1, mask_1 = midi_to_remi_tokens(file1, device=self.device)
        input_ids_2, mask_2 = midi_to_remi_tokens(file2, device=self.device)
        target = torch.tensor(label, dtype=torch.float, device=self.device)
        return input_ids_1, mask_1, input_ids_2, mask_2, target, file1, file2

def train(model, dataloader, optimizer, loss_fn, log_file):
    model.train()
    total_loss = 0
    log_print(f"🚀 Training 시작: {len(dataloader)} batch", log_file)
    for input_ids_1, mask_1, input_ids_2, mask_2, target, _, _ in tqdm(dataloader, desc="Training"):
        vec1 = model(input_ids_1.squeeze(1), mask_1.squeeze(1))
        vec2 = model(input_ids_2.squeeze(1), mask_2.squeeze(1))
        loss = loss_fn(vec1, vec2, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, log_file):
    model.eval()
    total_loss = 0
    log_print(f"Validation 시작: {len(dataloader)} batch", log_file)
    with torch.no_grad():
        for input_ids_1, mask_1, input_ids_2, mask_2, target, _, _ in tqdm(dataloader, desc="Validation"):
            vec1 = model(input_ids_1.squeeze(1), mask_1.squeeze(1))
            vec2 = model(input_ids_2.squeeze(1), mask_2.squeeze(1))
            loss = loss_fn(vec1, vec2, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, log_file, threshold=0.5, output_dir="results"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    total_TP = total_TN = total_FP = total_FN = 0
    file_idx = 1
    with torch.no_grad():
        for input_ids_1, mask_1, input_ids_2, mask_2, target, file1, file2 in tqdm(dataloader, desc="Evaluating"):
            vec1 = model(input_ids_1.squeeze(1), mask_1.squeeze(1))
            vec2 = model(input_ids_2.squeeze(1), mask_2.squeeze(1))
            cos_sim = F.cosine_similarity(vec1, vec2).item()
            pred = 1 if cos_sim >= threshold else -1
            true = target.item()
            TP = TN = FP = FN = 0
            if pred == 1 and true == 1: TP += 1
            elif pred == -1 and true == -1: TN += 1
            elif pred == 1 and true == -1: FP += 1
            elif pred == -1 and true == 1: FN += 1
            total_TP += TP
            total_TN += TN
            total_FP += FP
            total_FN += FN
            csv_file = os.path.join(output_dir, f"scores_eucli_{file_idx:05d}.csv")
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['File1', 'File2', 'CosineSim', 'Prediction', 'TrueLabel', 'TP', 'TN', 'FP', 'FN'])
                writer.writerow([file1[0], file2[0], cos_sim, pred, true, TP, TN, FP, FN])
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            log_print(f"{csv_file} ▶ 정확도: {accuracy*100:.2f}% (TP={TP}, TN={TN}, FP={FP}, FN={FN})", log_file)
            file_idx += 1
    total_acc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    log_print(f"\n[전체 정확도]\n▶ 정확도: {total_acc*100:.2f}% (TP={total_TP}, TN={total_TN}, FP={total_FP}, FN={total_FN})", log_file)
    return total_acc

device = torch.device("cpu")  
output_dir = r"D:\dayeon"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "training_log.txt")
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"📅 Training Log Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

model = MusicBERTEmbedding(vocab_size=5000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CosineEmbeddingLoss()

train_files = collect_midi_files(r'D:\02.라벨링데이터\VL', log_file)
val_files = collect_midi_files(r'D:\02.라벨링데이터\VL', log_file)
train_dataset = MIDISimilarityDataset(train_files, device=device)
val_dataset = MIDISimilarityDataset(val_files, device=device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 학습
epochs = 1
for epoch in range(epochs):
    log_print(f"\n🌀 Epoch {epoch+1}/{epochs} 시작: {datetime.now().strftime('%H:%M:%S')}", log_file)
    train_loss = train(model, train_loader, optimizer, loss_fn, log_file)
    val_loss = validate(model, val_loader, loss_fn, log_file)
    log_print(f" Epoch {epoch+1} 완료: {datetime.now().strftime('%H:%M:%S')}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", log_file)

log_print(f"\n🏁 Training 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
torch.save(model.state_dict(), os.path.join(output_dir, 'musicbert_model.pt'))
log_print(" 모델 저장 완료", log_file)

# 평가
evaluate(model, val_loader, log_file, threshold=0.5, output_dir=output_dir)
