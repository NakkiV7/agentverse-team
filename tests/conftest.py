"""Keep tests away from the player's local rating history."""
import pytest


@pytest.fixture(autouse=True)
def isolated_rating_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("BUDDY_PROFILE_FILE", str(tmp_path / "profile.json"))
    monkeypatch.setenv("BUDDY_PLAYER_ID", "test-player")
