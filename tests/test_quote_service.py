from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from io import BytesIO

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from api import app
from erp_tools import ERPError, load_erp_catalog
from quote_service import QuoteService, quote_workbook
from settings import DATA_DIR


def _decision(row, **fields):
    return {"item_id": row["item_id"], "action": "quote", **fields}


def test_reloads_checkpoint_and_current_inventory_then_generates_one_quote(tmp_path):
    catalog = load_erp_catalog()
    current = list(catalog)

    def loader():
        return tuple(current)

    path = tmp_path / "checkpoint.sqlite3"
    initial = QuoteService(path, catalog_loader=loader).start(
        source_name="手工询价", text="HX-259A 原装 10支\nHX-259A 原装 20支", mode="mock",
    )
    inquiry_id = initial.values["inquiry_id"]
    assert initial.next == ("human_review",)
    assert [row["status"] for row in initial.values["matched"]] == ["confirmed", "confirmed"]
    assert initial.values["elapsed_ms"].keys() >= {"load_file", "extract_rule", "extract_llm", "erp_match"}

    reopened = QuoteService(path, catalog_loader=loader)
    paused = reopened.get(inquiry_id)
    assert paused.interrupts[0].id == initial.interrupts[0].id
    assert paused.interrupts[0].value["source_text"]
    first, second = paused.values["matched"]

    sku_index = next(i for i, product in enumerate(current) if product.sku == "A001")
    current[sku_index] = replace(current[sku_index], stock=25, unit_price=Decimal("499.00"))
    rejected = reopened.resume(inquiry_id, [], interrupt_id=paused.interrupts[0].id)
    assert "合计 30" in rejected.interrupts[0].value["error"]
    assert "quote" not in rejected.values

    decisions = [_decision(second, quantity=15, reason="按当前库存调整")]
    stale = reopened.resume(inquiry_id, decisions, interrupt_id=paused.interrupts[0].id)
    assert stale.interrupts[0].id == rejected.interrupts[0].id
    assert "quote" not in stale.values

    def confirm(_):
        service = QuoteService(path, catalog_loader=loader)
        return service.resume(inquiry_id, decisions, interrupt_id=rejected.interrupts[0].id).values

    with ThreadPoolExecutor(max_workers=2) as pool:
        result, again = pool.map(confirm, range(2))
    assert result["quote_id"] == again["quote_id"] == f"Q-{inquiry_id}"
    assert result["audit"] == again["audit"]
    assert result["total"] == "12475.00"
    assert [row["ERP 可用库存"] for row in result["quote"]] == [25, 25]
    assert result["audit"][1]["审核备注"] == "按当前库存调整"
    assert result["audit"][1]["修改前"] != result["audit"][1]["修改后"]
    workbook = load_workbook(BytesIO(quote_workbook(result)))
    assert workbook.sheetnames == ["报价明细", "暂缓条目", "审核记录", "报价汇总"]
    assert "询价原型号" in [cell.value for cell in workbook["报价明细"][1]]


def test_erp_outage_then_unknown_model_and_explicit_postponement(tmp_path):
    def unavailable():
        raise ERPError("模拟 ERP 目录无法读取。")

    service = QuoteService(tmp_path / "checkpoint.sqlite3", catalog_loader=unavailable)
    initial = service.start(source_name="文本询价", text="LK-2441 80支\nZX-000 2盒")
    assert {entry["status"] for entry in initial.values["erp_calls"]} == {"error"}
    assert all("模拟 ERP 查询失败" in " ".join(row["issues"]) for row in initial.values["matched"])
    waiting = service.resume(initial.values["inquiry_id"], [], interrupt_id=initial.interrupts[0].id)
    assert "ERP 查询失败" in waiting.interrupts[0].value["error"]

    restored = QuoteService(service.db_path)
    actual = restored.start(source_name="文本询价", text="LK-2441 80支\nZX-000 2盒")
    first, second = actual.values["matched"]
    assert first["sku"] == "A004" and first["status"] == "need_check"
    assert second["status"] == "unrecognized"
    assert [call["status"] for call in actual.values["erp_calls"]] == ["ok", "not_found"]
    review = restored.resume(
        actual.values["inquiry_id"],
        [_decision(first, sku="A004", quantity=40, reason="按库存拆分需求"),
         {"item_id": second["item_id"], "action": "skip", "reason": "请客户提供铭牌型号"}],
        interrupt_id=actual.interrupts[0].id,
    )
    assert review.values["total"] == "10400.00"
    assert review.values["postponed"][0]["询价原型号"] == "ZX-000"
    assert review.values["audit"][1]["审核状态"] == "暂缓"


def test_api_excel_review_and_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_ENABLED", "0")
    monkeypatch.setenv("QUOTE_DB_PATH", str(tmp_path / "checkpoint.sqlite3"))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/assist", json={"text": "讲个笑话"}).json()["intent"] == "other"
    qa = client.post("/assist", json={"text": "HX-259A 原装多少钱？有多少库存？"}).json()
    assert qa["intent"] == "qa"
    assert [row["sku"] for row in qa["answer"]["products"]] == ["A001"]
    mixed = client.post("/assist", json={"text": "HX-259A 和 HX-259X 有什么区别？分别多少钱？"}).json()
    knowledge = mixed["answer"]["knowledge"]
    assert knowledge["quotes"] and knowledge["hits"][0]["chunk"]["source"] == "演示耗材选型指南.txt"
    text = client.post("/assist", json={"text": "请报价 HX-259A 原装 10支"}).json()
    assert text["intent"] == "quote" and text["quote"]["status"] == "waiting_review"
    direct_sku = client.post("/assist", json={"text": "A001 10支"}).json()["quote"]["items"][0]
    assert direct_sku["sku"] == "A001" and direct_sku["match_method"] == "精确 SKU"

    demo = DATA_DIR / "quotation_demo.xlsx"
    uploaded = client.post("/quotes/upload", files={"file": (demo.name, demo.read_bytes())})
    assert uploaded.status_code == 201
    draft = uploaded.json()
    inquiry_url = f"/quotes/{draft['inquiry_id']}"
    assert client.get(inquiry_url).json()["status"] == "waiting_review"
    assert client.get(f"{inquiry_url}/workbook").status_code == 409

    response = client.post(f"{inquiry_url}/review", json={"interrupt_id": draft["interrupt_id"], "decisions": []})
    rejected = response.json()
    assert rejected["review_error"] and rejected["status"] == "waiting_review"
    raw_rows = {row["source_row"]: row for row in rejected["items"] if row["status"] != "confirmed"}
    assert set(raw_rows) == {3, 4, 5, 6, 8}
    decisions = [
        _decision(raw_rows[3], sku="A003", quantity=5, reason="对照原文确认五个"),
        _decision(raw_rows[4], quantity=50, reason="按当前库存调整"),
        _decision(raw_rows[5], sku="A001", reason="选择标准容量"),
        {"item_id": raw_rows[6]["item_id"], "action": "skip", "reason": "请客户核对错写型号"},
        {"item_id": raw_rows[8]["item_id"], "action": "skip", "reason": "请客户提供原始型号"},
    ]
    approved = client.post(
        f"{inquiry_url}/review", json={"interrupt_id": rejected["interrupt_id"], "decisions": decisions},
    ).json()
    assert approved["status"] == "completed" and approved["quote_id"]
    assert approved["total"] == "32130.00"
    assert len(approved["postponed"]) == 2
    assert client.get(f"{inquiry_url}/workbook").content.startswith(b"PK")
