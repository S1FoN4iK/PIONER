"""Раздельное управление прокси: on / off / свой URL для каждого канала."""

from __future__ import annotations

import pytest

from conftest import settings

_P = "socks5://u:p@host:1080"


def test_defaults_route_telegram_and_downloads_only():
    s = settings(proxy_url=_P)
    assert s.telegram_proxy == _P
    assert s.download_proxy == _P
    assert s.ai_proxy is None


def test_no_proxy_url_means_direct_everywhere():
    s = settings()
    assert (s.telegram_proxy, s.download_proxy, s.ai_proxy) == (None, None, None)


def test_off_disables_single_channel():
    s = settings(proxy_url=_P, proxy_telegram="off")
    assert s.telegram_proxy is None
    assert s.download_proxy == _P


def test_ai_can_be_switched_on():
    s = settings(proxy_url=_P, proxy_ai="true")
    assert s.ai_proxy == _P


def test_channel_specific_url_overrides_common():
    own = "http://other:8080"
    s = settings(proxy_url=_P, proxy_downloads=own)
    assert s.download_proxy == own
    assert s.telegram_proxy == _P


def test_garbage_value_is_rejected():
    with pytest.raises(ValueError):
        settings(proxy_url=_P, proxy_telegram="maybe").telegram_proxy
