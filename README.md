# **Lab 5 – Part-of-Speech Tagging using RNN**
## **1. Implementation Steps**

Trong bài lab này, em xây dựng một mô hình RNN để gán nhãn từ loại (POS Tagging) trên
bộ dữ liệu UD English EWT. Dataset thầy cung cấp ở dạng JSONL, em tiền xử lý sang
dạng danh sách các câu gồm các cặp *(word, tag)* theo đúng yêu cầu bài lab.

Các bước triển khai:

1. **Tải và tiền xử lý dữ liệu**
   - Đọc ba file JSONL: train, dev, test.
   - Mỗi dòng chứa `"words"` và `"tags"` → chuyển thành danh sách
     `[(word, upos), ...]`.
   - Tách dữ liệu thành danh sách câu theo format mà bài lab yêu cầu.

2. **Xây dựng vocabulary**
   - Tạo `word_to_ix` với 2 token đặc biệt: `<UNK>`, `<PAD>`.
   - Tạo `tag_to_ix` với `<PAD>`.
   - In kích thước từ vựng và nhãn để kiểm tra.

3. **Xây dựng POSDataset và DataLoader**
   - Tạo class `POSDataset` kế thừa `torch.utils.data.Dataset`.
   - `__getitem__` trả về tensor index của từ và nhãn.
   - Viết `collate_fn` sử dụng `pad_sequence(batch_first=True)` để padding.
   - Tạo 3 DataLoader cho train, dev, test.

4. **Xây dựng mô hình RNN**
   - Mô hình gồm 3 tầng:
     - `nn.Embedding(vocab_size, emb_dim, padding_idx)`
     - `nn.RNN(emb_dim, hidden_dim, batch_first=True)`
     - `nn.Linear(hidden_dim, num_tags)`
   - Đầu ra của mô hình có shape *(batch_size, seq_len, num_tags)*.

5. **Huấn luyện mô hình**
   - Optimizer: **Adam (lr = 0.001)**
   - Loss function: **CrossEntropyLoss** với `ignore_index=pad_id`
   - Thực hiện vòng lặp 5 bước:
     1. Xóa gradient
     2. Forward
     3. Tính loss
     4. Backward
     5. Update
   - In loss + accuracy của train và dev mỗi epoch.

6. **Đánh giá mô hình**
   - Viết hàm `evaluate()`:
     - `model.eval()`, `torch.no_grad()`
     - Dùng `argmax` để lấy dự đoán
     - Chỉ tính accuracy trên những token không phải padding.
   - Thử dự đoán một câu mới bằng hàm `predict()`.

---

## **2. How to Run the Code**

Chạy toàn bộ notebook theo thứ tự:

1. Upload 3 file JSONL thầy cung cấp.
2. Chạy cell load dataset.
3. Chạy các cell xây dựng từ điển, dataset, dataloader.
4. Chạy cell model + training.
5. Chạy cell evaluate và predict.

Code chạy hoàn toàn không lỗi trên Google Colab.

---

## **3. Results Explanation**

### **Accuracy**
Sau 5 epoch, mô hình đạt được:

- **Train accuracy:** ~0.93  
- **Dev accuracy:** **0.8505**

Đây là kết quả hợp lý cho một mô hình RNN đơn giản (chưa dùng LSTM/BiLSTM).
Loss giảm đều qua từng epoch cho thấy mô hình học ổn định.

### **Prediction Example**
Với câu:
"I will get 10 points for this assignment."
Mô hình dự đoán:
I → PRON
will → AUX
get → VERB
10 → NUM
points → NOUN
for → ADP
this → DET
assignment. → NOUN
Kết quả hợp lý và phản ánh đúng POS cơ bản của tiếng Anh.

---

## **4. Difficulties and How I Solved Them**

- **Dataset thầy cung cấp không phải format .conllu như trong PDF**,  
  mà ở dạng JSONL.  
  ➜ Em đã tự điều chỉnh hàm đọc dữ liệu sang JSONL nhưng vẫn trả về đúng
  cấu trúc mà bài lab yêu cầu (`[(word, tag)]`).

- Padding phải xử lý cẩn thận để loss và accuracy không bị sai.  
  ➜ Em dùng `ignore_index` và tạo mask để loại bỏ token `<PAD>`.

- Một số câu rất dài, nếu batch size lớn dễ out of memory.  
  ➜ Em dùng batch_size = 32 để tối ưu tốc độ và bộ nhớ.

---

## **5. External Sources / Libraries**

- PyTorch (nn, optim, utils)
- Không sử dụng mô hình pre-trained
- Không sử dụng nguồn ngoài ngoài dataset thầy cung cấp

---

## **6. Final Report Values**

- **Dev Accuracy:** **0.8505**
- **Prediction Example:**
  - “I will get 10 points for this assignment.”
  - POS Tags: *PRON, AUX, VERB, NUM, NOUN, ADP, DET, NOUN*
