# Design document — EmbeddingGemma Code Search

Tài liệu này ghi lại **ý đồ thiết kế đã chốt** của skill semantic code search, để người mới đọc hiểu bối cảnh, ràng buộc, và hướng phát triển tiếp.

Ngôn ngữ chính của code/API: English.  
Ngôn ngữ giải thích thiết kế trong file này: Tiếng Việt.

---

## 1. Mục tiêu sản phẩm

Xây một **công cụ tìm code theo nghĩa** (semantic code search) phục vụ **agent coding** (Cursor, Claude Code, Antigravity, …), không thay thế `grep`/LSP mà bổ sung khả năng:

- Tìm theo hành vi / ý định: *“rate limiting”*, *“refresh token”*, *“auth middleware”*
- Trả về **đơn vị ngữ nghĩa nhỏ**: function / method / class (không chỉ cả file)
- Chạy được trong workflow agent: gọi script skill quen thuộc, không bắt buộc MCP server

**Model:** [Google EmbeddingGemma-2](https://huggingface.co/google/embeddinggemma-2) — embedding text/code, Matryoshka (768 / 512 / 256 / 128), prompt `CodeRetrieval` (query) và `Document` (chunk).

---

## 2. Vấn đề cần giải quyết (và quyết định)

### 2.1 Load model mỗi lần search → chậm

Mỗi lần `python scripts/search.py` là process mới → load model lại rất tốn thời gian.

**Quyết định:** Daemon host-level giữ model trong RAM.

| Hành vi | Chi tiết |
|---------|----------|
| Auto-start | Client không nối được socket → spawn daemon → chờ ready |
| Idle unload | Không request trong `EMBEDDINGGEMMA_IDLE_SEC` (mặc định 900s) → giải phóng model + thoát process |
| Queue ngắn | Nhiều agent gọi đồng thời → queue FIFO, `QUEUE_MAX` (mặc định 24); đầy thì trả `queue_full` |
| Đa repo | **Một model**, nhiều `index_dir` (index theo từng project) |
| UX skill | Agent vẫn chỉ gọi `python scripts/search.py "..."` — không cần biết daemon |

IPC: Unix domain socket (Linux/macOS); Windows / fallback: TCP localhost (`EMBEDDINGGEMMA_TCP=1`).

**Không** bắt buộc MCP/HTTP API cho luồng chính; daemon + socket là lớp “host shared process”. MCP vẫn có thể bọc lại sau nếu cần.

### 2.2 Va chạm ngữ nghĩa: code cũ vs code mới (agent đang sửa)

Nếu index tự cập nhật mọi file vừa edit giữa multi-step reasoning, agent dễ trộn snippet cũ (trong context) với index mới, hoặc search ra bản đã lỗi thời.

**Quyết định — chính sách reindex:**

```
Mỗi lần search (mặc định):
  → Chỉ reindex incremental khi git HEAD (commit) tăng so với last_indexed_commit
  → KHÔNG auto-reindex file dirty / working tree / agent vừa chạm

Agent muốn index đúng bản đang viết:
  → python scripts/index.py --paths <file1> <file2>
  → (tuỳ chọn) --working-tree nếu cần toàn bộ dirty
```

| Lớp | Nguồn chân lý | Ai cập nhật index |
|-----|----------------|-------------------|
| **Committed world** | Code đã commit | Auto khi search nếu commit đổi |
| **Working tree / touched** | File live trên đĩa | Agent chủ động `--paths` |

**Quy tắc tin cậy cho agent (bắt buộc ghi trong SKILL.md):**

1. Search = **discovery**, không phải source of truth sau khi đã edit.
2. Index mặc định bám **last committed**.
3. File vừa sửa → không auto vào index cho đến khi agent reindex.
4. Output search có thể liệt kê dirty files + gợi ý lệnh reindex.
5. Trước khi sửa logic quan trọng từ snippet → **đọc file live** tại path/line.

### 2.3 Hạt chunk: cả file vs AST

Chunk cả file → nhiễu, khó trúng đúng function.  
Chunk theo dòng cố định → cắt đứt ngữ nghĩa.

**Quyết định:** Tree-sitter AST, lấy node function / class / method / (một số type/trait…).

Ngôn ngữ hỗ trợ: **Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP** (+ fallback cả file / markdown).

Mỗi chunk mang metadata: `path`, `name`, `kind`, `parent`, `start_line`, `end_line`, `language`, `content`.

### 2.4 Skill vs MCP

| | Skill (hiện tại) | MCP server |
|--|------------------|------------|
| Tích hợp Cursor / Claude Code / Antigravity | Copy thư mục skill | Cấu hình MCP |
| Model sống lâu | Qua daemon + socket | Process MCP |
| Agent gọi | `python scripts/...` | Tool protocol |

**Quyết định:** Giao diện chính là **Skill** + script; daemon nằm dưới để tối ưu RAM/latency. Có thể thêm MCP wrapper sau mà không đổi chính sách index/chunk.

---

## 3. Kiến trúc tổng quan

```
┌─────────────┐  ┌─────────────┐
│ Agent A     │  │ Agent B     │   (Cursor / Claude / …)
│ search.py   │  │ search.py   │
└──────┬──────┘  └──────┬──────┘
       │  Unix socket / TCP
       └────────┬────────┘
                ▼
        ┌───────────────┐
        │  daemon.py    │  model load 1 lần, queue, idle shutdown
        │  + index LRU  │  cache vài index_dir gần đây
        └───────────────┘
                ▲
                │ đọc
        ┌───────────────┐
        │  .code_index/ │  documents.pkl, embeddings.npy, meta.json
        │  (theo repo)  │
        └───────────────┘
                ▲
        ┌───────────────┐
        │  index.py     │  commit-diff / --paths / --force / --working-tree
        │  chunker.py   │  tree-sitter
        └───────────────┘
```

### Vai trò từng module

| File | Trách nhiệm |
|------|-------------|
| `chunker.py` | Parse AST → list `CodeChunk` |
| `index.py` | Build/update index; policy commit vs paths vs working-tree |
| `search.py` | CLI: optional auto reindex commit-only → gọi daemon → in kết quả |
| `daemon.py` | Process nền: model, queue, search, idle |
| `client_ipc.py` | Auto-start daemon, gửi JSON request/response |
| `SKILL.md` | Hướng dẫn agent: khi nào search, khi nào reindex, quy tắc tin cậy |

### Index on disk (`.code_index/`)

- `documents.pkl` — list dict chunk + metadata  
- `embeddings.npy` — ma trận float32, L2-normalized  
- `meta.json` — `last_commit`, `file_hashes`, `dim`, `num_chunks`, `dirty_at_index`, …

Nên gitignore `.code_index/`.

---

## 4. Luồng vận hành chuẩn

### Search lần đầu trên máy (hoặc sau idle)

1. `search.py` thử socket → fail  
2. Spawn `daemon.py` → load EmbeddingGemma-2  
3. Reindex commit-only nếu cần / full nếu chưa có index  
4. Encode query (`CodeRetrieval`) → cosine top-k → in snippet  

### Search tiếp theo (trong cửa sổ idle)

Connect sẵn → (reindex commit nếu HEAD đổi) → encode query nhanh → kết quả.

### Agent vừa sửa file, cần search đúng bản mới

```bash
python scripts/index.py --paths src/foo.py src/bar.py
python scripts/search.py "..."
```

### Tắt daemon sớm / xem trạng thái

```bash
python scripts/daemon.py --status
python scripts/daemon.py --stop
```

---

## 5. Nguyên tắc không được phá khi phát triển tiếp

1. **Không** mặc định reindex mọi thay đổi working tree khi search.  
2. **Không** coi snippet index là source of truth sau edit — skill instruction phải giữ.  
3. **Không** bắt agent biết socket/daemon để search cơ bản.  
4. **Giữ** chunk theo AST (function/class…), không quay lại chỉ-chunk-file nếu không có lý do mạnh.  
5. **Giữ** tách: daemon = model + search; index policy = client/`index.py`.  
6. Model dùng chung host; index **không** trộn giữa các repo (`index_dir` riêng).

---

## 6. Hướng mở rộng gợi ý (chưa implement)

| Ý tưởng | Ghi chú |
|---------|--------|
| MCP server thin wrapper | Gọi cùng daemon/index; tool schema cho agent native MCP |
| Hybrid BM25 + dense | Tăng recall tên symbol chính xác |
| `touched_paths` log theo session | Agent/skill ghi path đã edit → output search cảnh báo rõ hơn |
| Hỗ trợ ngôn ngữ tree-sitter thêm | C#, Java, Kotlin… khi có binding ổn định |
| Index remote / CI artifact | Team share embedding đã build theo commit |
| Quantization / ONNX | Giảm RAM daemon trên máy yếu |
| Tests: chunker + policy reindex | Fixture đa ngôn ngữ + mock git |

Mọi mở rộng nên kiểm tra lại mục 5 (đặc biệt collision cũ–mới và auto-reindex).

---

## 7. Biến môi trường quan trọng

| Biến | Mặc định | Ý nghĩa |
|------|----------|--------|
| `CODE_DIR` | `.` | Root code cần index |
| `CODE_INDEX_DIR` | `./.code_index` | Thư mục index |
| `TRUNCATE_DIM` | `512` | Chiều embedding (Matryoshka) |
| `EMBEDDINGGEMMA_IDLE_SEC` | `900` | Idle rồi unload daemon |
| `EMBEDDINGGEMMA_SOCK` | `/tmp/embeddinggemma.sock` | Unix socket |
| `EMBEDDINGGEMMA_QUEUE_MAX` | `24` | Độ dài queue |
| `EMBEDDINGGEMMA_TCP` | `0` | `1` = TCP thay socket |

---

## 8. Bối cảnh repo

- Skill nằm trong monorepo **[Thinlt/Grok](https://github.com/Thinlt/Grok)** tại thư mục `embeddinggemma-code-search/`.  
- Repo standalone `Thinlt/embeddinggemma-code-search` đã được gỡ; **nguồn chính thức** là thư mục trong **Grok**.

---

## 9. Tóm tắt một câu

> Semantic search theo AST cho agent coding, model dùng chung qua daemon auto-start/idle, index ổn định theo git commit, file đang sửa do agent quyết định khi nào reindex — search để khám phá, file live mới là chân lý khi edit.

---

*Cập nhật theo các quyết định thiết kế đã thống nhất trong quá trình thiết kế skill (daemon, queue, commit-only reindex, agent-controlled touched files, tree-sitter multi-language, Skill-first).*
