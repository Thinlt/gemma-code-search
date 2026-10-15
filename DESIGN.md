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
  → KHÔNG tự resume index dở dang (backlog file chưa có hash) — chỉ cảnh báo [partial-index]

Agent muốn index đúng bản đang viết:
  → python scripts/index.py --paths <file1> <file2>
  → (tuỳ chọn) --working-tree nếu cần toàn bộ dirty

Index dở dang (Ctrl-C giữa chừng):
  → chạy lại python scripts/index.py (không --force) — resume từ checkpoint
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

### 2.5 Tuân theo .gitignore, dừng tương tác, hiển thị tiến độ

**.gitignore:** `collect_files` lọc file theo **đúng semantics git ignore** — gọi `git check-ignore -z --stdin` + `git ls-files --cached -z` theo **repo gần nhất** của từng file (hỗ trợ workspace multi-repo lồng nhau, nested `.gitignore`, `.git/info/exclude`, negation `!`, `**`); file đã tracked không bao giờ bị loại. File bị loại tự rời index ở lần persist kế tiếp (prune theo `all_files`). Thư mục non-git: không lọc (giữ hành vi cũ).

**Phím `q`:** khi index trong TTY, listener đọc keypress (cbreak) đặt cờ stop; vòng index check cờ giữa các file và giữa các sub-batch embed (64 chunks/lần). Khi dừng: flush sub-batch đang chạy → persist checkpoint → **exit code 0** (không coi là crash/error). Chạy lại lệnh cũ để resume từ checkpoint. Non-TTY (pipe, CI, agent shell) → listener không khởi động, hành vi y như cũ.

**Hiển thị tiến độ:** mỗi file đang index hiện trên một dòng rewrite (`\r`), nén dạng `<cấp1>/<cấp2>/...<cha ≤10 ký tự>/<file ≤32 ký tự>` — chỉ 2 cấp thư mục gốc, thu gọn các cấp giữa bằng `...`, hiện thư mục cha ngay trước file khi khác thư mục cấp 2, tên file quá 32 ký tự thì giữ đuôi (thấy extension) kèm tiền tố `...`.

**Phản hồi lúc embed (fix "đứng im sau Model ready"):** embed chạy theo sub-batch 64 chunk, mỗi batch mất chục giây không có file nào hoàn thành → ticker per-file đứng im, đọc như treo máy. Đo thực tế (probe wrapped 144 chunk thật): load model 7.6–9.6s, embed#1 7.0 ch/s vs embed#2 8.8 ch/s → warmup MPS chỉ ~1.3s (probe self-check đã warm sẵn), phần còn lại là compute thuần; máy bận thì tụt ~3 ch/s → 64 chunk im lặng 20–30s. Fix = ghi nhãn khoảng im lặng: mỗi sub-batch in `  [embed] N chunks ...` ngay trước khi encode và `done Xs (Y ch/s)` sau khi xong — không đổi tốc độ, chỉ hết cảm giác treo. Bổ sung sau khi thử thật: ticker per-file dùng `\r` trên một dòng và `close_progress()` xóa trắng dòng đó — chunking nhanh ~100× embed nên tên file chỉ loé lên <1s rồi bị xóa (như chưa từng hiện). Đổi `close_progress()` sang in xuống dòng: dòng file cuối cùng ở lại phía trên mỗi dòng `[embed]`, terminal đọc được nhịp `[N/M] path` + `[embed] ... done`.

### 2.6 Tốc độ embed: dtype và batch size cấu hình được, chọn theo đo đạc thực tế

Đo trên M2 Pro (MPS, torch 2.14, máy đang bận) với 24 chunk thật: **fp32 nhanh hơn bf16 ~18%** qua Metal, **fp16 ra embedding NaN** (Gemma tràn kích hoạt), **batch nhỏ hơn nhanh hơn** (bs=4 > bs=8 > bs=16 khi máy có tải khác). Do đó:

- `EMBEDDINGGEMMA_DTYPE=auto` (mặc định): CUDA giữ bf16 (đúng như cũ), MPS/CPU dùng fp32. Override trực tiếp bằng `fp32`/`bf16`/`fp16` (fp16 rủi ro NaN — có self-check tự hạ về fp32).
- `EMBEDDINGGEMMA_BATCH_SIZE` (mặc định 4): tăng 8/16 khi máy rảnh có thể nhanh hơn.
- Lưu ý: giảm `TRUNCATE_DIM` **không** làm index nhanh — encoder vẫn chạy đủ chiều, chỉ cắt vector cuối.
- **Cache embedding theo nội dung (sha1 full_text → vector)**: embedding là hàm thuần của (text, encoder) nên chunk giống hệt nhau tái dùng vector — sửa 1 hàm trong file 60 chunk chỉ re-embed 1 chunk, chạy lại sau `q` gần như miễn phí. Chỉ prime cache khi `meta.encoder` khớp encoder hiện tại (vector fp32 vs int8 không trộn). Đo thực tế trên log người dùng: 1.6 ch/s × ~1600 token/chunk template lớn ≈ 2.5k token/s — bằng đúng tốc độ token của lần đo rảnh 8.8 ch/s × ~250 token → máy không chậm, workload template thật sự có 15–16× token/chunk. Token/giây là hằng số → mọi tối ưu phải giảm SỐ TOKEN, không phải "tăng tốc".
- **Bỏ `file: {path}` khỏi full_text (đã làm, text_format 2)**: đo trên một corpus lớn gồm phần lớn là template view sinh ra từ cùng khuôn (nhiều bản sao chỉ khác path): hit-rate cache theo full_text = 0%, nhưng theo content (bỏ path) = 31% + 12% trùng trong chính đợt — các template giống hệt nhau nhưng path khác → vector khác. Format mới `title: … | lang: … | text: …` mở khóa dedup giữa các file; path vẫn tới search qua BM25 (corpus BM25 gồm `path + name + content`) và key `"path"` lưu trong documents. **A/B kiểm chứng trước khi đổi** (22 file / 388 chunk, cùng corpus build 2 index A=cũ/B=bỏ path, 10 query thật qua daemon hybrid): hit@1 7/10 cả hai, MRR@5 0.720 vs 0.700 — khác biệt duy nhất hòa hoàn toàn ở k=20 (#4/#7 giống hệt); 2 query miss **giống nhau ở cả hai** → lỗi có sẵn của template-vs-query, không phải do bỏ path.
- **Migration an toàn bằng `meta.text_format`** (index.py, `TEXT_FORMAT = 2`): vector cũ embed chuỗi khác nên chỉ được **rebuild, không bao giờ merge**. Guard gom cả dim-mismatch lẫn text_format: (1) run full/incremental thường → in `[index-format] … rebuilding from scratch` rồi reset index như nhánh dim cũ; (2) `--paths`/`--subdir`/`--commit-only` → **abort exit 1** với chỉ dẫn `index.py --force` (update theo scope sẽ trộn vector 2 định dạng hoặc co index về scope); (3) search auto-reindex (chính là `--commit-only`) → in `[error] … full rebuild first` ra stderr nhưng **vẫn trả kết quả** từ index cũ — không bao giờ tự kick off rebuild 5k file trong một lần search. Cache embedding không bao giờ tái dùng vector xuyên định dạng vì khóa sha1(full_text) đổi theo chuỗi mới.

### 2.7 Hybrid BM25 + dense: recall tên symbol chính xác

Dense embedding mạnh về ngữ nghĩa nhưng yếu khi query chứa **tên symbol chính xác** (`AuthService.login`, `RetryPolicy.maxAttempts`) — token hiếm bị nhòa trong vector. BM25 bù đúng điểm yếu này. Thiết kế:

- **Tokenize nhận diện identifier**: tách camelCase/PascalCase (`refreshTokenRotation` → `refresh`, `token`, `rotation`) **và** giữ nguyên dạng lowercase nguyên khối (`refreshtokenrotation`) — khớp được cả query viết kiểu người lẫn kiểu tên hàm.
- **Corpus BM25** = tokens của `path + name + content` mỗi chunk; index BM25 build lười theo `mtime_ns` của `documents.pkl`, nằm trong cùng LRU cache index của daemon (không tốn gì khi rank_bm25 chưa cài → tự tắt hybrid, phản hồi vẫn có `hybrid_used: false`).
- **Fusion RRF** (k=60): dense pool (quét theo score giảm dần, ngưng dưới `min_score`, lấy `max(50, top_k*8)` ứng viên) + BM25 pool (score > 0, cùng cỡ) → `1/(k+rank)` cộng dồn hai bảng → xếp hạng cuối. Per-file cap áp **sau** fusion nên không cắt hụt ứng viên tốt.
- **Đo thực tế** (trên một codebase production): query là tên method `"verifyDocumentFormat"` — hybrid đẩy method cùng tầng, khớp token, lên #2; dense-only lấp #2–#4 bằng các anonymous function gần đúng về mặt typography. Query mô tả bằng lời `"document validation flow"` — hybrid đưa interface/contract tương ứng lên; dense-only chỉ ra hooks.
- **Tắt theo query**: `search.py --no-bm25` (request `"hybrid": false`).

### 2.8 Backend ONNX int8: opt-in cho máy không có GPU torch

Mục tiêu ban đầu: giảm RAM / tăng tốc. Kết quả đo trên M2 Pro lại **chậm hơn** (int8 CPU EP 37.8s vs torch fp32 MPS 15.6s cho 48 chunk) → backend mặc định giữ `torch`; ONNX là opt-in (`EMBEDDINGGEMMA_BACKEND=onnx`) hợp lý trên máy CPU-only hoặc muốn tiết kiệm RAM.

Điểm kỹ thuật đã xử lý:

- **Export**: `torch.onnx.export(dynamo=True)` (fallback legacy) qua wrapper `_TextOnly` ép keyword-arg — model gốc multimodal, forward vị trí sẽ nuốt `attention_mask` vào `pixel_values`. Dynamo externalize trọng lượng fp32 ra `model.onnx.data`; bản int8 tự chứa (0 external tensor) nên script dọn cả hai file fp32 sau quantize.
- **Quantize**: `quantize_dynamic(QInt8, op_types_to_quantize=["MatMul"])` — **bỏ qua** bảng embedding 262k token (op Gather): quantize bảng này kéo cosine xuống dưới ngưỡng.
- **Pooling phải gồm token prompt**: đây là bug khó nhất. Cấu hình pooling ghi `include_prompt: true` nhưng thực đo ST 6.1 với model này: processor path **không bao giờ set `prompt_length`**, nên mean pooling tính trên **mọi token có attention mask, kể cả prompt**. Probe brute-force các cửa sổ loại-prompt: loại 0 token → cosine 1.0015 (≈ đúng tuyệt đối, lệch >1 do drift MPS-vs-CPU). Trước khi sửa: cos_mean 0.965–0.969 (fail ngưỡng 0.97); sau: **0.9732**.
- **Validation tự chặn**: export so sánh với ST fp32 trên chunk thật từ index lokal (cosine từng chunk + overlap top-5 của 6 query) với gate mean ≥ 0.97 / min ≥ 0.90 / overlap ≥ 0.60; fail thì **xóa bundle** và exit 1 — không bao giờ để bundle kém chất lượng tự kích hoạt.
- **Execution provider**: mặc định CPU EP. CoreML EP chia graph này thành ~290 partition (2083/3563 node hỗ trợ) → partition-thrash + OOM khi compile lúc máy bận; chỉ bật qua `EMBEDDINGGEMMA_ONNX_EP=coreml` sau khi tự đo.
- **Nhất quán encoder**: `meta.json` ghi trường `encoder` (vd `embeddinggemma-2:torch:float32:512` hay `embeddinggemma-2:onnx-int8:512`); daemon so với encoder hiện tại, lệch thì cảnh báo `_encoder mismatch_` + gợi ý `index.py --force`. Cosine giữa encoder khác nhau không so được — index int8 + query fp32 cho score vô nghĩa.

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
| `daemon.py` | Process nền: model, queue, search, idle, BM25 hybrid fusion |
| `client_ipc.py` | Auto-start daemon, gửi JSON request/response |
| `onnx_embedder.py` | Backend int8 ONNX: replicate pipeline ST (prompt → tokenize → pool → slice → normalize) |
| `export_onnx.py` | Export → quantize → validate → bundle int8 (tự xóa khi fail gate) |
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

1. **Không** mặc định reindex mọi thay đổi working tree khi search, và **không** tự resume backlog index dở dang khi search.  
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
| ~~Hybrid BM25 + dense~~ ✅ 2.7 | RRF k=60, tokenizer camelCase, `--no-bm25` tắt theo query |
| `touched_paths` log theo session | Agent/skill ghi path đã edit → output search cảnh báo rõ hơn |
| Hỗ trợ ngôn ngữ tree-sitter thêm | C#, Java, Kotlin… khi có binding ổn định |
| Index remote / CI artifact | Team share embedding đã build theo commit |
| ~~Quantization / ONNX~~ ✅ 2.8 | int8 MatMul-only, opt-in `EMBEDDINGGEMMA_BACKEND=onnx`; chậm hơn torch trên máy GPU |
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
| `EMBEDDINGGEMMA_BATCH_SIZE` | `4` | Batch size khi embed (tăng khi máy rảnh) |
| `EMBEDDINGGEMMA_DTYPE` | `auto` | `fp32`/`bf16`/`fp16`; auto = bf16 trên CUDA, fp32 trên MPS/CPU |
| `EMBEDDINGGEMMA_BACKEND` | `torch` | `torch`/`onnx`; onnx cần bundle int8 đã export (§2.8) |
| `EMBEDDINGGEMMA_ONNX_EP` | `auto` | Execution provider ONNX; mặc định CPU EP (CoreML chia partition nặng) |
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
