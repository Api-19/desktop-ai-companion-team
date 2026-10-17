from pathlib import Path
from typing import Protocol

from companion.importer.models import ImportBatch


class ChatImportAdapter(Protocol):
    def parse(self, path: Path) -> ImportBatch: ...
