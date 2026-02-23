"""Structural validation of k8s manifests and docker-compose.yml."""

from pathlib import Path

import pytest
import yaml

K8S_DIR = Path("k8s")
COMPOSE_FILE = Path("docker-compose.yml")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> list[dict]:
    """Load one or more YAML documents from a file."""
    docs = list(yaml.safe_load_all(path.read_text()))
    return [d for d in docs if d is not None]


def _all_manifests() -> list[tuple[str, dict]]:
    """Collect (relative_path, parsed_doc) for every YAML in k8s/."""
    result = []
    for path in sorted(K8S_DIR.rglob("*.yaml")):
        for doc in _load_yaml(path):
            result.append((str(path), doc))
    return result


# ---------------------------------------------------------------------------
# Generic manifest requirements
# ---------------------------------------------------------------------------


class TestManifestStructure:
    @pytest.mark.parametrize("path,doc", _all_manifests())
    def test_has_api_version(self, path, doc):
        assert "apiVersion" in doc, f"{path}: missing apiVersion"

    @pytest.mark.parametrize("path,doc", _all_manifests())
    def test_has_kind(self, path, doc):
        assert "kind" in doc, f"{path}: missing kind"

    @pytest.mark.parametrize("path,doc", _all_manifests())
    def test_has_metadata_name(self, path, doc):
        assert "name" in doc.get("metadata", {}), f"{path}: missing metadata.name"

    @pytest.mark.parametrize("path,doc", _all_manifests())
    def test_namespace_is_visionops(self, path, doc):
        kind = doc.get("kind", "")
        # Namespace resource itself defines the namespace — skip
        if kind == "Namespace":
            return
        ns = doc.get("metadata", {}).get("namespace")
        assert ns == "visionops", f"{path}: namespace should be 'visionops', got {ns!r}"


# ---------------------------------------------------------------------------
# Inference deployment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inference_deployment() -> dict:
    docs = _load_yaml(K8S_DIR / "inference" / "deployment.yaml")
    return next(d for d in docs if d.get("kind") == "Deployment")


@pytest.fixture(scope="module")
def inference_service() -> dict:
    docs = _load_yaml(K8S_DIR / "inference" / "service.yaml")
    return next(d for d in docs if d.get("kind") == "Service")


@pytest.fixture(scope="module")
def inference_hpa() -> dict:
    docs = _load_yaml(K8S_DIR / "inference" / "hpa.yaml")
    return next(d for d in docs if d.get("kind") == "HorizontalPodAutoscaler")


class TestInferenceDeployment:
    def test_container_port_is_8000(self, inference_deployment):
        container = inference_deployment["spec"]["template"]["spec"]["containers"][0]
        ports = [p["containerPort"] for p in container.get("ports", [])]
        assert 8000 in ports

    def test_liveness_probe_path(self, inference_deployment):
        container = inference_deployment["spec"]["template"]["spec"]["containers"][0]
        path = container["livenessProbe"]["httpGet"]["path"]
        assert path == "/health"

    def test_readiness_probe_path(self, inference_deployment):
        container = inference_deployment["spec"]["template"]["spec"]["containers"][0]
        path = container["readinessProbe"]["httpGet"]["path"]
        assert path == "/health"

    def test_resource_limits_set(self, inference_deployment):
        container = inference_deployment["spec"]["template"]["spec"]["containers"][0]
        assert "limits" in container["resources"]
        assert "requests" in container["resources"]

    def test_configmap_ref_present(self, inference_deployment):
        container = inference_deployment["spec"]["template"]["spec"]["containers"][0]
        refs = [e["configMapRef"]["name"] for e in container.get("envFrom", [])]
        assert "inference-config" in refs

    def test_selector_matches_pod_labels(self, inference_deployment):
        selector = inference_deployment["spec"]["selector"]["matchLabels"]
        pod_labels = inference_deployment["spec"]["template"]["metadata"]["labels"]
        for key, val in selector.items():
            assert pod_labels.get(key) == val


class TestInferenceService:
    def test_selector_matches_deployment_label(self, inference_service):
        assert inference_service["spec"]["selector"]["app"] == "inference"

    def test_target_port_is_8000(self, inference_service):
        ports = inference_service["spec"]["ports"]
        assert any(p["targetPort"] == 8000 for p in ports)


class TestInferenceHPA:
    def test_targets_inference_deployment(self, inference_hpa):
        ref = inference_hpa["spec"]["scaleTargetRef"]
        assert ref["kind"] == "Deployment"
        assert ref["name"] == "inference"

    def test_max_replicas(self, inference_hpa):
        assert inference_hpa["spec"]["maxReplicas"] >= 2

    def test_cpu_metric_configured(self, inference_hpa):
        metrics = inference_hpa["spec"]["metrics"]
        cpu = next((m for m in metrics if m["resource"]["name"] == "cpu"), None)
        assert cpu is not None


# ---------------------------------------------------------------------------
# MLflow manifests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlflow_deployment() -> dict:
    docs = _load_yaml(K8S_DIR / "mlflow" / "deployment.yaml")
    return next(d for d in docs if d.get("kind") == "Deployment")


@pytest.fixture(scope="module")
def mlflow_service() -> dict:
    docs = _load_yaml(K8S_DIR / "mlflow" / "service.yaml")
    return next(d for d in docs if d.get("kind") == "Service")


class TestMLflowDeployment:
    def test_container_port_is_5000(self, mlflow_deployment):
        container = mlflow_deployment["spec"]["template"]["spec"]["containers"][0]
        ports = [p["containerPort"] for p in container.get("ports", [])]
        assert 5000 in ports

    def test_pvc_mounted(self, mlflow_deployment):
        spec = mlflow_deployment["spec"]["template"]["spec"]
        pvc_volumes = [
            v for v in spec.get("volumes", []) if "persistentVolumeClaim" in v
        ]
        assert len(pvc_volumes) >= 1

    def test_selector_matches_pod_labels(self, mlflow_deployment):
        selector = mlflow_deployment["spec"]["selector"]["matchLabels"]
        pod_labels = mlflow_deployment["spec"]["template"]["metadata"]["labels"]
        for key, val in selector.items():
            assert pod_labels.get(key) == val


class TestMLflowService:
    def test_selector_matches_deployment_label(self, mlflow_service):
        assert mlflow_service["spec"]["selector"]["app"] == "mlflow"

    def test_port_is_5000(self, mlflow_service):
        ports = mlflow_service["spec"]["ports"]
        assert any(p["port"] == 5000 for p in ports)


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


class TestDockerCompose:
    def test_has_inference_service(self, compose):
        assert "inference" in compose["services"]

    def test_has_mlflow_service(self, compose):
        assert "mlflow" in compose["services"]

    def test_inference_depends_on_mlflow(self, compose):
        deps = compose["services"]["inference"].get("depends_on", {})
        assert "mlflow" in deps

    def test_inference_port_8000(self, compose):
        ports = compose["services"]["inference"]["ports"]
        assert any("8000" in str(p) for p in ports)

    def test_mlflow_port_5000(self, compose):
        ports = compose["services"]["mlflow"]["ports"]
        assert any("5000" in str(p) for p in ports)

    def test_mlflow_volume_defined(self, compose):
        assert "mlflow-data" in compose.get("volumes", {})
