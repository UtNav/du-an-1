import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ===== 1. Đọc dữ liệu từ CSV =====
def load_data_csv(file_path, corrected_path="hypernymy_data_large_corrected.csv"): 
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f: 
            first_line = f.readline().strip().lower() # Đọc dòng đầu tiên để kiểm tra header
        has_header = "label" in first_line and "word" in first_line
        df = pd.read_csv(file_path, header=0 if has_header else None, encoding="utf-8-sig")
        if len(df.columns) != 4:
            raise ValueError(" File CSV không có đúng 4 cột.")
        if not has_header:
            df.columns = ["word1", "word2", "label", "relation"]
        df["label"] = df["label"].astype(int)
        df.to_csv(corrected_path, index=False, encoding="utf-8-sig")
        print(f" Đã chuẩn hóa và lưu file: {corrected_path}")
        return df
    except Exception as e:
        print(" Lỗi khi đọc CSV:", e)
        exit(1)

# ===== 2. Tính trước embedding PhoBERT =====
def compute_embeddings(df, tokenizer, encoder, device, batch_size=32): # Tính embedding cho từng từ trong DataFrame
    print(" Đang tính embedding theo batch với PhoBERT...")
    encoder.eval() # Chuyển encoder sang chế độ eval để tắt dropout

    word_list = list(set(df['word1'].tolist() + df['word2'].tolist())) # Lấy tất cả từ duy nhất
    embeddings = {}

    for i in range(0, len(word_list), batch_size): 
        batch_words = word_list[i:i + batch_size] # Chia thành batch
        inputs = tokenizer(batch_words, return_tensors="pt", truncation=True, padding=True, max_length=128) # Chuyển đổi từ thành tensor
        inputs = {k: v.to(device) for k, v in inputs.items()} # Chuyển tensor sang thiết bị (GPU/CPU)
        with torch.no_grad():
            outputs = encoder(**inputs) # Lấy embedding từ encoder
        vecs = outputs.last_hidden_state[:, 0, :].cpu() # Lấy embedding của từ (CLS token)
        for word, vec in zip(batch_words, vecs):
            embeddings[word] = vec # vec là embedding của từ

    print(" Đã tính xong embedding.")
    return embeddings # embeddings là từ điển {từ: embedding}

# ===== 3. Dataset từ embedding đã tính =====
def create_dataset_from_embeddings(df, embedding_dict):
    x1, x2, y = [], [], [] # Lưu trữ embedding của từ 1, từ 2 và nhãn
    for _, row in df.iterrows(): # Lặp qua từng dòng của DataFrame
        e1 = embedding_dict[row["word1"]] # Lấy embedding của từ 1
        e2 = embedding_dict[row["word2"]] # Lấy embedding của từ 2
        x1.append(e1) # Lưu embedding của từ 1
        x2.append(e2) # Lưu embedding của từ 2
        y.append(row["label"]) # Lưu nhãn
    return TensorDataset(torch.stack(x1), torch.stack(x2), torch.tensor(y)) # Tạo TensorDataset từ embedding và nhãn

# ===== 4. Mô hình =====
class HypernymyClassifier(nn.Module):
    def __init__(self, input_dim=768*2, hidden_dim=256, output_dim=2): # input_dim là kích thước embedding của PhoBERT
        super().__init__() # Khởi tạo mô hình
        self.fc1 = nn.Linear(input_dim, hidden_dim) # Fully connected layer đầu tiên
        self.relu = nn.ReLU() # Hàm kích hoạt ReLU
        self.dropout = nn.Dropout(0.3) # Dropout để tránh overfitting
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Fully connected layer thứ hai

    def forward(self, x1, x2):# Kết hợp embedding của hai từ
        x = torch.cat((x1, x2), dim=1) # Nối embedding của hai từ
        x = self.fc1(x) # Chuyển qua layer đầu tiên
        x = self.relu(x) # Áp dụng ReLU
        x = self.dropout(x) # Áp dụng dropout
        return self.fc2(x) # Chuyển qua layer cuối cùng để dự đoán nhãn

# ===== 5. Huấn luyện =====
def train_model(model, dataloader, optimizer, criterion, device): # Huấn luyện mô hình
    model.train() # Chuyển mô hình sang chế độ huấn luyện
    for epoch in range(5): # Lặp qua 5 epoch
        total_loss = 0 # Biến để lưu tổng loss
        for x1, x2, y in dataloader: # Lặp qua từng batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device) # Chuyển dữ liệu sang thiết bị (GPU/CPU)
            optimizer.zero_grad() # Đặt gradient về 0
            out = model(x1, x2) # Dự đoán nhãn
            loss = criterion(out, y) # Tính loss
            loss.backward() # Tính gradient
            optimizer.step()    # Cập nhật trọng số
            total_loss += loss.item() # Cộng dồn loss
        print(f" Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}") # In loss trung bình của epoch

# ===== 6. Đánh giá =====
def evaluate_model(model, dataset, device): # Đánh giá mô hình trên tập dữ liệu
    model.eval() # Chuyển mô hình sang chế độ eval
    y_true, y_pred = [], []     # Biến để lưu nhãn thực tế và dự đoán
    with torch.no_grad(): # Tắt gradient để tiết kiệm bộ nhớ
        for x1, x2, y in DataLoader(dataset, batch_size=32): # Lặp qua từng batch
            x1, x2 = x1.to(device), x2.to(device) # Chuyển dữ liệu sang thiết bị
            out = model(x1, x2) # Dự đoán nhãn
            y = y.to(device) # Chuyển nhãn sang thiết bị
            pred = torch.argmax(out, dim=1).cpu() # Lấy nhãn dự đoán
            y_true.extend(y.tolist()) # Lưu nhãn thực tế
            y_pred.extend(pred.tolist()) # Lưu nhãn dự đoán
    acc = accuracy_score(y_true, y_pred) # Tính accuracy
    f1 = f1_score(y_true, y_pred) # Tính F1-score
    cm = confusion_matrix(y_true, y_pred) # Tính confusion matrix
    print("\n Đánh giá:")   # In kết quả đánh giá
    print(f" Accuracy: {acc:.4f}") # In accuracy
    print(f" F1-Score: {f1:.4f}") # In F1-score
    print(" Confusion Matrix:\n", cm) # In confusion matrix

# ===== 7. Dự đoán =====
def predict(model, tokenizer, encoder, word1, word2, device): # Dự đoán quan hệ bao thuộc giữa hai từ
    model.eval() # Chuyển mô hình sang chế độ eval
    inputs = tokenizer([word1, word2], return_tensors="pt", truncation=True, padding=True, max_length=128).to(device) # Chuyển đổi từ thành tensor
    with torch.no_grad(): # Tắt gradient để tiết kiệm bộ nhớ
        outputs = encoder(**inputs) # Lấy embedding từ encoder
    emb1 = outputs.last_hidden_state[0, 0, :].unsqueeze(0).to(device) # Lấy embedding của từ 1 (CLS token)
    emb2 = outputs.last_hidden_state[1, 0, :].unsqueeze(0).to(device) # Lấy embedding của từ 2 (CLS token)
    with torch.no_grad(): # Dự đoán quan hệ bao thuộc
        out = model(emb1, emb2) # Dự đoán nhãn
        pred = torch.argmax(out, dim=1).item() # Lấy nhãn dự đoán
    return " Có quan hệ bao thuộc" if pred == 1 else " Không có quan hệ bao thuộc", pred # Trả về kết quả dự đoán và nhãn

# ===== MAIN =====
if __name__ == "__main__": # Main function to run the script
    FILE_PATH = "hypernymy_data_large.csv" # Đường dẫn đến file CSV gốc
    CORRECTED_FILE = "hypernymy_data_large_corrected.csv" # Đường dẫn đến file CSV đã chuẩn hóa
    MODEL_PATH = "phobert_hypernymy.pt"

    print(" Kiểm tra thiết bị...") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Kiểm tra xem có GPU hay không
    print(" Sử dụng:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    df = load_data_csv(FILE_PATH, CORRECTED_FILE) # Đọc dữ liệu từ file CSV
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # Chia dữ liệu thành tập huấn luyện và kiểm tra

    model_name = "vinai/phobert-base" # Sử dụng PhoBERT base model
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Tải tokenizer của PhoBERT
    encoder = AutoModel.from_pretrained(model_name).to(device) # Tải encoder của PhoBERT và chuyển sang thiết bị

    model = HypernymyClassifier().to(device) # Khởi tạo mô hình
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Sử dụng Adam optimizer
    criterion = nn.CrossEntropyLoss() # Sử dụng CrossEntropyLoss làm hàm mất mát

    if os.path.exists(MODEL_PATH): # Kiểm tra xem mô hình đã được huấn luyện chưa
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # Tải mô hình đã huấn luyện
        model.eval() #
        print(f"Đã tải mô hình từ {MODEL_PATH}") 
    else:
        print(" Đang tính embedding và huấn luyện mô hình...") 
        full_df = pd.concat([train_df, test_df], ignore_index=True) 
        embedding_dict = compute_embeddings(full_df, tokenizer, encoder, device) 
        train_dataset = create_dataset_from_embeddings(train_df, embedding_dict) 
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
        train_model(model, train_loader, optimizer, criterion, device) 
        torch.save(model.state_dict(), MODEL_PATH) 
        print(f" Đã lưu mô hình vào: {MODEL_PATH}")

    test_embedding = compute_embeddings(test_df, tokenizer, encoder, device) #  Tính embedding cho tập kiểm tra
    test_dataset = create_dataset_from_embeddings(test_df, test_embedding) # Tạo dataset từ embedding
    evaluate_model(model, test_dataset, device) # Đánh giá mô hình trên tập kiểm tra

    # ==== Dự đoán tương tác ====
    while True:
        print("\n Dự đoán quan hệ bao thuộc:")
        w1 = input(" Từ 1 (hoặc 'exit'): ").strip() 
        if w1.lower() == "exit":
            break
        w2 = input(" Từ 2: ").strip()
        result, pred = predict(model, tokenizer, encoder, w1, w2, device) # Dự đoán quan hệ bao thuộc
        print(" Kết quả:", result)

        save = input(" Lưu cặp này vào file? (y/n): ").strip().lower() # Lưu cặp từ vào file
        if save == "y":
            relation_type = input(" Loại quan hệ (hypernym, random, co-hypernym): ").strip() 
            with open(CORRECTED_FILE, "a", encoding="utf-8-sig") as f:
                f.write(f"{w1},{w2},{pred},{relation_type}\n")
            print(" Đã lưu.")
