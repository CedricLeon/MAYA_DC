import os

from omegaconf import OmegaConf

from src.train import _configure_single_node_ddp_environment


def _make_cfg(
    *,
    strategy: str = "ddp",
    accelerator: str = "gpu",
    devices=4,
    num_nodes: int = 1,
):
    return OmegaConf.create(
        {
            "trainer": {
                "strategy": strategy,
                "accelerator": accelerator,
                "devices": devices,
                "num_nodes": num_nodes,
            }
        }
    )


def test_single_node_ddp_env_defaults_to_loopback(monkeypatch) -> None:
    for key in ("MASTER_ADDR", "MASTER_PORT", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        monkeypatch.delenv(key, raising=False)

    _configure_single_node_ddp_environment(_make_cfg())

    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"
    assert os.environ["MASTER_PORT"].isdigit()


def test_single_node_ddp_env_overrides_inherited_socket_interfaces(monkeypatch) -> None:
    monkeypatch.setenv("MASTER_ADDR", "10.10.10.10")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ib0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ib0")

    _configure_single_node_ddp_environment(_make_cfg())

    assert os.environ["MASTER_ADDR"] == "10.10.10.10"
    assert os.environ["MASTER_PORT"] == "23456"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "lo"
    assert os.environ["GLOO_SOCKET_IFNAME"] == "lo"


def test_single_node_ddp_env_skips_non_ddp_or_single_device(monkeypatch) -> None:
    for key in ("MASTER_ADDR", "MASTER_PORT", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        monkeypatch.delenv(key, raising=False)

    _configure_single_node_ddp_environment(_make_cfg(strategy="auto"))
    assert "MASTER_ADDR" not in os.environ

    _configure_single_node_ddp_environment(_make_cfg(devices=1))
    assert "MASTER_ADDR" not in os.environ

    _configure_single_node_ddp_environment(_make_cfg(num_nodes=2))
    assert "MASTER_ADDR" not in os.environ
