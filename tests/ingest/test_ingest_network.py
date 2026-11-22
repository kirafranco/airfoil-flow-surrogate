"""Opt-in check against the real host: `pytest -m network`.

Confirms the archive the pipeline was written against is still the one being
served. A failure here means the source changed, which needs a human decision.
"""

from __future__ import annotations

import pytest

from src.ingest.airfrans import AIRFRANS_URL, RELEASE_ETAG, RELEASE_SIZE_BYTES
from src.ingest.download import head


@pytest.mark.network
def test_the_remote_archive_is_the_known_release() -> None:
    remote = head(AIRFRANS_URL)
    assert remote.etag == RELEASE_ETAG
    assert remote.size_bytes == RELEASE_SIZE_BYTES
    assert remote.last_modified is not None and "11 Aug 2022" in remote.last_modified
