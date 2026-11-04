#!/usr/bin/env python3
"""
Import-time noise control for the embedding stack.

`import sentence_transformers` transitively imports
`torch.distributed.elastic.multiprocessing.redirects`, whose module body calls
`get_libc()` and logs a warning on macOS:

    NOTE: Redirects are currently not supported in MacOs.

That code redirects stdout/stderr at the file-descriptor level for
`torchrun`/torchelastic workers, which this project never uses — the message is
purely informational and arrives before any of our own output is set up, so it
reads as if something went wrong. Raising that one logger above WARNING keeps
runs clean without hiding anything actionable.

No third-party imports here, so the cost of importing this module is nil.
"""

from __future__ import annotations

import logging

# Set before the first `import sentence_transformers` / `import torch.distributed.elastic...`
_REDIRECTS_LOGGER = "torch.distributed.elastic.multiprocessing.redirects"


def silence_import_warnings() -> None:
    """Mute the known-harmless warnings emitted while importing the model stack."""
    logging.getLogger(_REDIRECTS_LOGGER).setLevel(logging.ERROR)
