#!/usr/bin/env python3
"""
AST-aware code chunker using tree-sitter.
Supports: Python, JavaScript, JSX, TypeScript, TSX, Go, Rust, PHP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Language, Node, Parser


def _load_languages() -> Dict[str, Language]:
    langs: Dict[str, Language] = {}

    try:
        import tree_sitter_python as tsp

        langs["python"] = Language(tsp.language())
    except Exception:
        pass

    try:
        import tree_sitter_javascript as tsjs

        langs["javascript"] = Language(tsjs.language())
        langs["jsx"] = langs["javascript"]
    except Exception:
        pass

    try:
        import tree_sitter_typescript as tsts

        langs["typescript"] = Language(tsts.language_typescript())
        langs["tsx"] = Language(tsts.language_tsx())
    except Exception:
        pass

    try:
        import tree_sitter_go as tsgo

        langs["go"] = Language(tsgo.language())
    except Exception:
        pass

    try:
        import tree_sitter_rust as tsr

        langs["rust"] = Language(tsr.language())
    except Exception:
        pass

    try:
        import tree_sitter_php as tsphp

        if hasattr(tsphp, "language_php"):
            langs["php"] = Language(tsphp.language_php())
        else:
            langs["php"] = Language(tsphp.language())
    except Exception:
        pass

    return langs


LANGUAGES = _load_languages()

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
}

TARGET_TYPES = {
    "function_definition",
    "class_definition",
    "async_function_definition",
    "function_declaration",
    "method_definition",
    "class_declaration",
    "arrow_function",
    "generator_function_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    "method_declaration",
    "type_declaration",
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "mod_item",
    "trait_declaration",
}

# Production split budget in tokens (title + signature context counted in).
# Measured on the real corpus: long chunks split into ~320-token parts with a
# signature context line retrieve tail-of-method content far better
# (hit@5 18% -> 45% with per-source-chunk result merging), and the size also
# fits fixed-320 ANE embedding backends. Subsumes the old MAX_CHUNK_CHARS
# guard — a 320-token part can never request the 100+ GiB tensors that
# froze the host on whole-file chunks (minified JS, base64, embedded assets).
# Bump TEXT_FORMAT in index.py when changing this or full_text.
TOKEN_LIMIT = 320

_TOKENIZER: Any = None
_TOKENIZER_TRIED = False


def _count_tokens(texts: List[str]) -> List[int]:
    """Token counts from the embedder's tokenizer (HF cache); chars/3.5 est
    when transformers or the cache is unavailable — parts may then overshoot
    slightly, which only matters for strict-320 backends, not for MPS/CPU."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        try:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
        except Exception:
            _TOKENIZER = None
    if _TOKENIZER is not None:
        return [len(x) for x in _TOKENIZER(texts, add_special_tokens=True)["input_ids"]]
    return [max(1, int(len(t.encode()) / 3.5)) for t in texts]


@dataclass
class CodeChunk:
    path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    content: str
    parent: Optional[str] = None
    language: str = ""
    part: Optional[Tuple[int, int]] = None  # (i, n) when split from a longer chunk
    context_sig: str = ""  # signature line of the original chunk, prefixed on parts 2+

    @property
    def full_text(self) -> str:
        # no `file:` segment: path tokens drag near-duplicate files (122 template
        # near-copies) apart in vector space and block embedding-cache reuse, while
        # carrying no meaning a query would use — the path still reaches search via
        # BM25 and the stored "path" key. Bump TEXT_FORMAT in index.py when changing.
        name = f"{self.parent}.{self.name}" if self.parent else self.name
        title = f"{self.kind}: {name}"
        if self.part:
            # parts of one source chunk share name/parent — search merges them
            # by those fields and shows (part i/n) so the split is visible
            title = f"{self.kind}(part {self.part[0]}/{self.part[1]}): {name}"
        body = self.content
        if self.part and self.context_sig and not body.lstrip().startswith(self.context_sig[:20]):
            # signature context: later parts keep "what function is this" even
            # though their content starts mid-body
            body = f"context: {self.context_sig}\n{body}"
        return f"title: {title} | lang: {self.language} | text: {body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "parent": self.parent,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "language": self.language,
            "part": self.part,
            "full_text": self.full_text,
        }


def _node_text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_name(node: Node, source: bytes) -> str:
    for child in node.children:
        if child.type in (
            "identifier",
            "name",
            "property_identifier",
            "type_identifier",
        ):
            return _node_text(source, child)
        if child.type in ("function_declarator", "method_definition"):
            return _find_name(child, source)
    return "anonymous"


def _classify(node_type: str) -> str:
    if any(k in node_type for k in ("class", "struct", "interface", "trait", "enum")):
        return "class"
    if "method" in node_type:
        return "method"
    if any(k in node_type for k in ("impl", "mod")):
        return "block"
    return "function"


def _walk(
    node: Node,
    source: bytes,
    path: str,
    lang: str,
    parent_name: Optional[str] = None,
) -> List[CodeChunk]:
    chunks: List[CodeChunk] = []

    if node.type in TARGET_TYPES:
        name = _find_name(node, source)
        if name == "anonymous":
            # disambiguate by start position: distinct anonymous chunks (e.g.
            # arrow functions assigned to props) must not share a name, or the
            # search-time part merge would stitch unrelated functions together
            name = f"anonymous@{node.start_point[0] + 1}:{node.start_point[1]}"
        content = _node_text(source, node).strip()
        if content and len(content) > 10:
            kind = _classify(node.type)
            chunks.append(
                CodeChunk(
                    path=path,
                    name=name,
                    kind=kind,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    content=content,
                    parent=parent_name,
                    language=lang,
                )
            )
            new_parent = name if kind in ("class", "block") else parent_name
            for child in node.children:
                chunks.extend(_walk(child, source, path, lang, new_parent))
            return chunks

    for child in node.children:
        chunks.extend(_walk(child, source, path, lang, parent_name))
    return chunks


def _signature_line(content: str) -> str:
    """First non-empty line, capped — for a method/class chunk this is its
    declaration line; parts 2+ carry it as their `context:` prefix."""
    for ln in content.splitlines():
        s = ln.strip()
        if s:
            return s[:90]
    return ""


def _split_oversized(chunks: List[CodeChunk]) -> List[CodeChunk]:
    lens = _count_tokens([c.full_text for c in chunks])
    out: List[CodeChunk] = []
    for c, total_tokens in zip(chunks, lens):
        if total_tokens <= TOKEN_LIMIT:
            out.append(c)
            continue
        sig = _signature_line(c.content)
        # reserve tokens for the widest part title, the context prefix, slack
        head = _count_tokens([
            f"title: {c.kind}(part 99/99): {(c.parent + '.' if c.parent else '') + c.name}"
            f" | lang: {c.language} | text: "
        ])[0]
        sig_toks = _count_tokens([f"context: {sig}\n"])[0] if sig else 0
        budget = TOKEN_LIMIT - head - sig_toks - 8
        parts = _content_parts(c.content, budget)
        n = len(parts)
        for i, (text, line_offset) in enumerate(parts, 1):
            ck = CodeChunk(
                path=c.path,
                name=c.name,
                kind=c.kind,
                start_line=c.start_line + line_offset,
                end_line=c.start_line + line_offset + text.count("\n"),
                content=text,
                parent=c.parent,
                language=c.language,
                part=(i, n),
                context_sig=sig,
            )
            # the chars/token estimate can undershoot; trim whole body lines
            # until the real tokenizer confirms the final full_text fits
            while _count_tokens([ck.full_text])[0] > TOKEN_LIMIT and ck.content.count("\n") > 1:
                ck.content = ck.content[: ck.content.rfind("\n")]
            # dense single lines (base64, minified JS) can tokenize near
            # 1 token/char — 3x the estimate — so fall back to char cuts
            while _count_tokens([ck.full_text])[0] > TOKEN_LIMIT and len(ck.content) > 64:
                ck.content = ck.content[: max(64, int(len(ck.content) * 0.85))]
            ck.end_line = ck.start_line + ck.content.count("\n")
            out.append(ck)
    return out


def _content_parts(content: str, budget_toks: int) -> List[Tuple[str, int]]:
    """Split content into ~budget_toks parts at line boundaries (est per line,
    verified by the caller); single monster lines (minified JS, base64) are
    hard-cut at budget chars — a line can never straddle a part boundary."""
    parts: List[Tuple[str, int]] = []
    cur: List[str] = []
    cur_t = 0
    offset = 0  # chars consumed so far, for start_line math
    part_offset = 0

    def flush() -> None:
        nonlocal cur, cur_t, part_offset
        if cur:
            parts.append(("".join(cur), content.count("\n", 0, part_offset)))
            cur, cur_t = [], 0

    for ln in content.splitlines(keepends=True):
        est = max(1, int(len(ln.encode()) / 3.7))
        if est > budget_toks:  # monster line: hard-cut by chars
            flush()
            step = max(200, budget_toks * 3)
            for k in range(0, len(ln), step):
                parts.append((ln[k : k + step], content.count("\n", 0, offset)))
            offset += len(ln)
            part_offset = offset
            continue
        if cur and cur_t + est > budget_toks:
            flush()
            part_offset = offset
        cur.append(ln)
        cur_t += est
        offset += len(ln)
    flush()
    return parts


def chunk_source(source: str, path: str, language: str) -> List[CodeChunk]:
    if language not in LANGUAGES:
        lines = source.count("\n") + 1
        chunks = [
            CodeChunk(
                path=path,
                name=Path(path).name,
                kind="file",
                start_line=1,
                end_line=lines,
                content=source,
                language=language or "unknown",
            )
        ]
        return _split_oversized(chunks)

    parser = Parser(LANGUAGES[language])
    src_bytes = bytes(source, "utf-8")
    tree = parser.parse(src_bytes)
    chunks = _walk(tree.root_node, src_bytes, path, language)

    if not chunks:
        lines = source.count("\n") + 1
        chunks = [
            CodeChunk(
                path=path,
                name=Path(path).name,
                kind="file",
                start_line=1,
                end_line=lines,
                content=source,
                language=language,
            )
        ]
    return _split_oversized(chunks)


def chunk_file(path: str, content: str) -> List[CodeChunk]:
    ext = Path(path).suffix.lower()
    lang = EXT_TO_LANG.get(ext, "")
    return chunk_source(content, path, lang)
