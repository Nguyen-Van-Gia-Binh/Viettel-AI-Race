# -*- coding: utf-8 -*-
"""Unit tests for the local LLM integration (client, extraction logic, and routing)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clinlp.llm.client import LLMClient
from clinlp.ner.llm_extract import _align_casing_with_source, llm_ner_fn
from clinlp.ner.pipeline_adapter import ner_fn
from clinlp.schema import Assertion, ConceptType


def test_align_casing_with_source():
    source = "Bệnh nhân Ho khan kéo dài."
    # Extracted text has different casing
    extracted = "ho khan"
    aligned = _align_casing_with_source(source, extracted)
    assert aligned == "Ho khan"

    # If it is not in the source, should return the extracted text as-is
    assert _align_casing_with_source(source, "sốt") == "sốt"


@patch("httpx.Client.post")
def test_llm_client_chat_completion(mock_post):
    # Mock httpx response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "test response"
                }
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    client = LLMClient()
    response = client.chat_completion([{"role": "user", "content": "hi"}])
    
    assert response == "test response"
    mock_post.assert_called_once()


@patch("clinlp.llm.client.LLMClient.chat_completion")
def test_llm_ner_fn(mock_chat):
    # Mock LLM JSON output mimicking semantic parsing
    mock_chat.return_value = json.dumps({
        "bệnh_sử": {
            "triệu_chứng": [
                {
                    "tên": "ho",
                    "ngữ_cảnh": "phủ định"
                }
            ]
        },
        "tiền_sử": {
            "chẩn_đoán": [
                {
                    "tên": "tăng huyết áp"
                }
            ]
        },
        "xét_nghiệm": [
            {
                "tên": "Troponin",
                "kết_quả": "0.01",
                "đơn_vị": "ng/mL"
            }
        ]
    })

    source_text = "Bệnh nhân có Tiền sử tăng huyết áp nhưng không ho. Xét nghiệm Troponin 0.01 ng/mL."
    results = list(llm_ner_fn(source_text))

    # Should yield 4 concepts: ho, tăng huyết áp, Troponin, and 0.01
    assert len(results) == 4
    
    # Extract concept details by name
    ho_mention = next(r for r in results if r[0].lower() == "ho")
    assert ho_mention[1] == ConceptType.TRIEU_CHUNG
    assert Assertion.NEGATED in ho_mention[2]
    
    ht_mention = next(r for r in results if r[0].lower() == "tăng huyết áp")
    assert ht_mention[1] == ConceptType.CHAN_DOAN
    assert Assertion.HISTORICAL in ht_mention[2]

    trop_mention = next(r for r in results if r[0].lower() == "troponin")
    assert trop_mention[1] == ConceptType.TEN_XET_NGHIEM
    assert not trop_mention[2]

    res_mention = next(r for r in results if r[0] == "0.01")
    assert res_mention[1] == ConceptType.KET_QUA_XET_NGHIEM
    assert not res_mention[2]


@patch("clinlp.ner.llm_extract.llm_ner_fn")
@patch("clinlp.ner.pipeline_adapter.CONFIG_PATH")
def test_pipeline_adapter_routing(mock_config_path, mock_llm_ner_fn):
    from unittest.mock import mock_open
    config_data = json.dumps({
        "pipeline": {
            "mode": "llm"
        }
    })
    mock_config_path.exists.return_value = True
    
    # Mock llm_ner_fn return
    mock_llm_ner_fn.return_value = [("ho", ConceptType.TRIEU_CHUNG, [Assertion.NEGATED])]

    with patch("builtins.open", mock_open(read_data=config_data)):
        results = list(ner_fn("Bệnh nhân không ho."))
        
    assert len(results) == 1
    assert results[0][0] == "ho"
    assert Assertion.NEGATED in results[0][2]
    mock_llm_ner_fn.assert_called_once()
