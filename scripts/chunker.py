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

# Keep chunks within the embedder's token budget (~1.5-2k tokens); whole-file
# chunks (minified JS, template files, base64 assets) can otherwise request
# 100+ GiB tensors inside the model and freeze the host.
MAX_CHUNK_CHARS = 6000


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

    @property
    def full_text(self) -> str:
        title = f"{self.kind}: {self.name}"
        if self.parent:
            title = f"{self.kind}: {self.parent}.{self.name}"
        return (
            f"title: {title} | file: {self.path} | "
            f"lang: {self.language} | text: {self.content}"
        )

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


def _split_oversized(chunks: List[CodeChunk]) -> List[CodeChunk]:
    out: List[CodeChunk] = []
    for c in chunks:
        if len(c.content) <= MAX_CHUNK_CHARS:
            out.append(c)
            continue
        parts = _content_parts(c.content)
        total = len(parts)
        for i, (text, line_offset) in enumerate(parts, 1):
            out.append(
                CodeChunk(
                    path=c.path,
                    name=f"{c.name} (part {i}/{total})",
                    kind=c.kind,
                    start_line=c.start_line + line_offset,
                    end_line=c.start_line + line_offset + text.count("\n"),
                    content=text,
                    parent=c.parent,
                    language=c.language,
                )
            )
    return out


def _content_parts(content: str) -> List[Tuple[str, int]]:
    parts: List[Tuple[str, int]] = []
    idx = 0
    while idx < len(content):
        end = min(idx + MAX_CHUNK_CHARS, len(content))
        if end < len(content):
            nl = content.find("\n", end)
            # snap to the next line boundary only when it is close; a single long
            # line (minified JS, base64) must be hard-cut instead of swallowed
            if nl != -1 and nl - end <= MAX_CHUNK_CHARS // 6:
                end = nl + 1
        parts.append((content[idx:end], content.count("\n", 0, idx)))
        idx = end
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
