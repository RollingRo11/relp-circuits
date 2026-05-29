"""Smoke tests that don't require a GPU or model download."""

import importlib


def test_imports():
    importlib.import_module("relp_circuits")
    importlib.import_module("relp_circuits.model")
    importlib.import_module("relp_circuits.tasks.sva")
    importlib.import_module("relp_circuits.attribution")
    importlib.import_module("relp_circuits.attribution.ig")
    importlib.import_module("relp_circuits.attribution.relp")
    importlib.import_module("relp_circuits.ablation")
