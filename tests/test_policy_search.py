"""真实政策检索适配测试。"""

from core import policy_search


def test_official_policy_search_normalizes_results(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "resultCode": {"code": 200},
                "result": {
                    "data": {
                        "middle": {
                            "list": [
                                {
                                    "title": "<em>农机</em>购置补贴政策",
                                    "summary": "支持符合条件的农业经营主体。",
                                    "url": "https://www.gov.cn/policy/example.htm",
                                    "time": "2026-01-02",
                                }
                            ]
                        }
                    }
                },
            }

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json})
        return FakeResponse()

    monkeypatch.setattr(policy_search.requests, "post", fake_post)
    monkeypatch.setattr(policy_search, "_official_headers", lambda: {})

    results = policy_search.search_official_policies("农机补贴")

    assert results == [
        {
            "title": "农机购置补贴政策",
            "summary": "支持符合条件的农业经营主体。",
            "url": "https://www.gov.cn/policy/example.htm",
            "source": "中国政府网",
            "published_at": "2026-01-02",
        }
    ]
    assert captured["json"]["dataTypeId"] == "14"
    assert captured["json"]["searchBy"] == "all"
