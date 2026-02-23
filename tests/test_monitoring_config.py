"""Validation of Prometheus config, Grafana provisioning, and k8s monitoring manifests."""

import json
from pathlib import Path

import pytest
import yaml

PROMETHEUS_YML = Path("configs/prometheus/prometheus.yml")
GRAFANA_DATASOURCE = Path("configs/grafana/provisioning/datasources/prometheus.yaml")
GRAFANA_DASHBOARDS_PROV = Path(
    "configs/grafana/provisioning/dashboards/dashboards.yaml"
)
GRAFANA_DASHBOARD_JSON = Path("configs/grafana/dashboards/visionops.json")
K8S_MONITORING_DIR = Path("k8s/monitoring")
COMPOSE_FILE = Path("docker-compose.yml")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _k8s_manifests() -> list[tuple[str, dict]]:
    result = []
    for path in sorted(K8S_MONITORING_DIR.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc is not None:
                result.append((str(path), doc))
    return result


# ---------------------------------------------------------------------------
# Prometheus config
# ---------------------------------------------------------------------------


class TestPrometheusConfig:
    @pytest.fixture(scope="class")
    def cfg(self) -> dict:
        return _load_yaml(PROMETHEUS_YML)

    def test_has_global_scrape_interval(self, cfg):
        assert "scrape_interval" in cfg["global"]

    def test_has_scrape_configs(self, cfg):
        assert "scrape_configs" in cfg
        assert len(cfg["scrape_configs"]) >= 1

    def test_has_inference_job(self, cfg):
        jobs = [s["job_name"] for s in cfg["scrape_configs"]]
        assert "inference" in jobs

    def test_inference_metrics_path(self, cfg):
        job = next(s for s in cfg["scrape_configs"] if s["job_name"] == "inference")
        assert job.get("metrics_path", "/metrics") == "/metrics"

    def test_inference_has_target(self, cfg):
        job = next(s for s in cfg["scrape_configs"] if s["job_name"] == "inference")
        targets = job["static_configs"][0]["targets"]
        assert len(targets) >= 1


# ---------------------------------------------------------------------------
# Grafana datasource provisioning
# ---------------------------------------------------------------------------


class TestGrafanaDatasource:
    @pytest.fixture(scope="class")
    def cfg(self) -> dict:
        return _load_yaml(GRAFANA_DATASOURCE)

    def test_api_version_present(self, cfg):
        assert cfg["apiVersion"] == 1

    def test_has_prometheus_datasource(self, cfg):
        names = [ds["name"] for ds in cfg["datasources"]]
        assert "Prometheus" in names

    def test_datasource_type_is_prometheus(self, cfg):
        ds = next(d for d in cfg["datasources"] if d["name"] == "Prometheus")
        assert ds["type"] == "prometheus"

    def test_datasource_points_to_prometheus_service(self, cfg):
        ds = next(d for d in cfg["datasources"] if d["name"] == "Prometheus")
        assert "prometheus" in ds["url"].lower()

    def test_datasource_is_default(self, cfg):
        ds = next(d for d in cfg["datasources"] if d["name"] == "Prometheus")
        assert ds.get("isDefault") is True


# ---------------------------------------------------------------------------
# Grafana dashboard provisioning
# ---------------------------------------------------------------------------


class TestGrafanaDashboardProvisioning:
    @pytest.fixture(scope="class")
    def cfg(self) -> dict:
        return _load_yaml(GRAFANA_DASHBOARDS_PROV)

    def test_has_providers(self, cfg):
        assert "providers" in cfg
        assert len(cfg["providers"]) >= 1

    def test_provider_type_is_file(self, cfg):
        assert cfg["providers"][0]["type"] == "file"

    def test_provider_path_set(self, cfg):
        assert "path" in cfg["providers"][0]["options"]


# ---------------------------------------------------------------------------
# Grafana dashboard JSON
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(GRAFANA_DASHBOARD_JSON.read_text())


class TestGrafanaDashboard:
    def test_uid_is_set(self, dashboard):
        assert dashboard.get("uid") == "visionops-inference"

    def test_has_panels(self, dashboard):
        assert len(dashboard["panels"]) >= 3

    def test_has_predictions_total_query(self, dashboard):
        all_exprs = [
            t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])
        ]
        assert any("predictions_total" in e for e in all_exprs)

    def test_has_errors_query(self, dashboard):
        all_exprs = [
            t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])
        ]
        assert any("prediction_errors_total" in e for e in all_exprs)

    def test_has_latency_query(self, dashboard):
        all_exprs = [
            t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])
        ]
        assert any("inference_latency_seconds_bucket" in e for e in all_exprs)

    def test_has_p95_latency(self, dashboard):
        all_exprs = [
            t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])
        ]
        assert any("0.95" in e for e in all_exprs)

    def test_refresh_is_set(self, dashboard):
        assert dashboard.get("refresh")

    def test_has_visionops_tag(self, dashboard):
        assert "visionops" in dashboard.get("tags", [])


# ---------------------------------------------------------------------------
# k8s monitoring manifests — generic structure
# ---------------------------------------------------------------------------


class TestK8sMonitoringManifests:
    @pytest.mark.parametrize("path,doc", _k8s_manifests())
    def test_has_api_version(self, path, doc):
        assert "apiVersion" in doc, f"{path}: missing apiVersion"

    @pytest.mark.parametrize("path,doc", _k8s_manifests())
    def test_has_kind(self, path, doc):
        assert "kind" in doc, f"{path}: missing kind"

    @pytest.mark.parametrize("path,doc", _k8s_manifests())
    def test_namespace_is_visionops(self, path, doc):
        ns = doc.get("metadata", {}).get("namespace")
        assert ns == "visionops", f"{path}: expected namespace visionops, got {ns!r}"

    def test_prometheus_configmap_contains_scrape_config(self):
        cm = _load_yaml(K8S_MONITORING_DIR / "prometheus-configmap.yaml")
        data = cm["data"]["prometheus.yml"]
        cfg = yaml.safe_load(data)
        jobs = [s["job_name"] for s in cfg["scrape_configs"]]
        assert "inference" in jobs

    def test_prometheus_configmap_uses_k8s_dns(self):
        cm = _load_yaml(K8S_MONITORING_DIR / "prometheus-configmap.yaml")
        data = cm["data"]["prometheus.yml"]
        assert "svc.cluster.local" in data

    def test_prometheus_deployment_mounts_configmap(self):
        dep = _load_yaml(K8S_MONITORING_DIR / "prometheus-deployment.yaml")
        volumes = dep["spec"]["template"]["spec"]["volumes"]
        cm_vols = [v for v in volumes if "configMap" in v]
        assert any(v["configMap"]["name"] == "prometheus-config" for v in cm_vols)

    def test_grafana_deployment_port_is_3000(self):
        dep = _load_yaml(K8S_MONITORING_DIR / "grafana-deployment.yaml")
        container = dep["spec"]["template"]["spec"]["containers"][0]
        ports = [p["containerPort"] for p in container.get("ports", [])]
        assert 3000 in ports

    def test_grafana_service_port_is_3000(self):
        svc = _load_yaml(K8S_MONITORING_DIR / "grafana-service.yaml")
        assert any(p["port"] == 3000 for p in svc["spec"]["ports"])


# ---------------------------------------------------------------------------
# docker-compose — monitoring services added
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


class TestDockerComposeMonitoring:
    def test_has_prometheus_service(self, compose):
        assert "prometheus" in compose["services"]

    def test_has_grafana_service(self, compose):
        assert "grafana" in compose["services"]

    def test_prometheus_port_9090(self, compose):
        ports = compose["services"]["prometheus"]["ports"]
        assert any("9090" in str(p) for p in ports)

    def test_grafana_port_3000(self, compose):
        ports = compose["services"]["grafana"]["ports"]
        assert any("3000" in str(p) for p in ports)

    def test_prometheus_mounts_config(self, compose):
        volumes = compose["services"]["prometheus"]["volumes"]
        assert any("prometheus.yml" in str(v) for v in volumes)

    def test_grafana_mounts_provisioning(self, compose):
        volumes = compose["services"]["grafana"]["volumes"]
        assert any("provisioning" in str(v) for v in volumes)

    def test_grafana_depends_on_prometheus(self, compose):
        deps = compose["services"]["grafana"].get("depends_on", [])
        assert "prometheus" in deps
