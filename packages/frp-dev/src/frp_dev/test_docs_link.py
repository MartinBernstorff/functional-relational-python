from pathlib import Path

import pytest
from safe_result import Err, Ok

from frp_dev.docs_link import (
    Checkout,
    DocsLinked,
    DocsLinker,
    DocsLinkOccupiedError,
    DocsSource,
    DocsSourceMissingError,
    DocsUnlinked,
)


@pytest.fixture
def source(tmp_path: Path) -> DocsSource:
    root = tmp_path / "docs-source"
    root.mkdir()
    return DocsSource(root)


@pytest.fixture
def checkout(tmp_path: Path) -> Checkout:
    root = tmp_path / "checkout"
    root.mkdir()
    return Checkout(root)


def test_link_creates_a_symlink_to_the_source(
    source: DocsSource, checkout: Checkout
) -> None:
    result = DocsLinker.link(source, checkout)

    assert result == Ok(DocsLinked.CREATED)
    assert checkout.docs_link().resolve() == source.root.resolve()


def test_link_is_idempotent(source: DocsSource, checkout: Checkout) -> None:
    _ = DocsLinker.link(source, checkout)

    result = DocsLinker.link(source, checkout)

    assert result == Ok(DocsLinked.ALREADY_PRESENT)


def test_link_reports_a_missing_source(tmp_path: Path, checkout: Checkout) -> None:
    missing = DocsSource(tmp_path / "missing")

    result = DocsLinker.link(missing, checkout)

    match result:
        case Err(DocsSourceMissingError() as error):
            assert error.source == missing
        case _:
            pytest.fail(f"expected DocsSourceMissingError, got {result}")
    assert not checkout.docs_link().is_symlink()


def test_link_leaves_an_existing_docs_folder_alone(
    source: DocsSource, checkout: Checkout
) -> None:
    checkout.docs_link().mkdir()

    result = DocsLinker.link(source, checkout)

    match result:
        case Err(DocsLinkOccupiedError() as error):
            assert error.checkout == checkout
        case _:
            pytest.fail(f"expected DocsLinkOccupiedError, got {result}")
    assert checkout.docs_link().is_dir()
    assert not checkout.docs_link().is_symlink()


def test_unlink_removes_the_symlink_but_not_the_source(
    source: DocsSource, checkout: Checkout
) -> None:
    _ = DocsLinker.link(source, checkout)

    outcome = DocsLinker.unlink(checkout)

    assert outcome == DocsUnlinked.REMOVED
    assert not checkout.docs_link().exists()
    assert source.root.is_dir()


def test_unlink_without_a_symlink_does_nothing(checkout: Checkout) -> None:
    assert DocsLinker.unlink(checkout) == DocsUnlinked.NOT_PRESENT
