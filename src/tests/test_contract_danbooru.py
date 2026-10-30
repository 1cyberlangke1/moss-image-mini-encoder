"""接口契约测试：把「真实服务的硬限制」钉死，而不是 mock 掉喵。

背景：`test_source_danbooru.py` 里的分页测试全部 mock 了 `_fetch_page`，
所以它只能验证我自己的二分逻辑，验证不了「服务端第 1001 页返回 410」这种
真实约束。今天正是因为这条约束没被测到，才出现抓到 100 万条就卡死的事故喵。

这类测试要联网、较慢，默认跳过；手动跑：
    $env:RUN_NETWORK_TESTS=1
    pytest src/tests/test_contract_danbooru.py -o pythonpath=src -q
"""

from __future__ import annotations

import os
import urllib.error

import pytest

from moss_small_encoder.sources import danbooru

NETWORK = os.environ.get("RUN_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not NETWORK, reason="需要联网；设 RUN_NETWORK_TESTS=1 开启契约测试"
)


def _session():
    """建一个走本地代理的 opener喵。"""
    return danbooru._opener("http://127.0.0.1:7890")


def test_page_1000_may_fail_at_the_edge() -> None:
    """第 1000 页在边界上不稳定：实测返回 500，不是 410喵。

    输入: 无
    输出: 断言要么成功，要么抛 500
    预期: 记录「边界内侧已经算不动」这个事实；
          它说明分片必须显著小于 100 万，不能贴着上限切喵
    """
    try:
        rows = danbooru._fetch_page(_session(), 1000)
    except urllib.error.HTTPError as exc:
        assert exc.code in (410, 500)
        return
    assert len(rows) <= danbooru.PAGE_LIMIT


def test_page_1001_is_gone() -> None:
    """第 1001 页返回 410 Gone，这是硬上限喵。

    输入: 无
    输出: 断言抛 HTTPError 且 code == 410
    预期: 一旦服务端改成别的行为，这条测试立刻报警喵
    """
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        danbooru._fetch_page(_session(), 1001)
    assert excinfo.value.code == 410


def test_id_range_bypasses_the_page_cap() -> None:
    """按 id 区间查同一个深层位置仍然可用，证明区间是绕开上限的正确手段喵。

    输入: 无
    输出: 断言区间查询第 1 页非空且 id 落在区间内
    预期: 这是「必须按 id 分片」这条设计的事实依据喵
    """
    rows = danbooru._fetch_page(_session(), 1, 2_000_000, 3_000_000)
    assert rows
    assert all(2_000_000 <= row["id"] <= 3_000_000 for row in rows)
