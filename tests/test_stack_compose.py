"""The project's container standards, asserted as rules against servicios/.

These tests never start Docker. They read the compose file, the Dockerfiles and
the configuration templates, and check that what is written there obeys the
standards: bind mounts only, pinned official base images, profiles, resource
limits, the GPU reserved for training only, bind addresses as settings, and no
secret in any versioned file.
"""

from __future__ import annotations

import configparser
import os
import re
from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT

SERVICIOS = PROJECT_ROOT / "servicios"
COMPOSE_PATH = SERVICIOS / "docker-compose.yml"
ENV_EXAMPLE = SERVICIOS / ".env.example"
ENV_FILE = SERVICIOS / ".env"

#: Resource limits per service: 4 GB and 2 CPUs for Jupyter and for ML training.
LIMITS = {
    "ml": {"memory": "4G", "cpus": "2"},
    "jupyter": {"memory": "4G", "cpus": "2"},
}

#: Nothing starts without a profile.
PROFILES = {"ml": ["ml"], "jupyter": ["dev"]}

#: The job runner keeps no state of its own: it writes into the shared data
#: tree, never into ../data/ml.
STATELESS = {"ml"}

#: Jupyter writes notebooks/, because writing notebooks is what it is for.
WRITABLE_OWN_STATE = {"jupyter": {"../notebooks"}}

RESTART_POLICIES = {"unless-stopped", "always"}
#: Declared exceptions to the restart rule, with the value each declares.
#: `ml` is a run-to-completion job: restarting it would re-run finished work.
RESTART_EXCEPTIONS = {"ml": "no"}

#: Only training holds the GPU: a 6 GB card has no VRAM to share.
GPU_SERVICES = {"ml"}

SECRET_KEY = re.compile(r"PASSWORD|SECRET|TOKEN")
VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
BASE_IMAGE = re.compile(r"python:\d+\.\d+\.\d+-slim")
PIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+-]*")

#: Never walked when scanning for leaked secrets: data is not versioned, and
#: the data tree holds tens of gigabytes of CFD snapshots.
SKIP_DIRS = {"data", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def env_values(path: Path) -> dict[str, str]:
    return {
        line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.strip().startswith("#")
    }


def dockerfile_path(service: dict) -> Path:
    build = service["build"]
    return (SERVICIOS / build["context"] / build["dockerfile"]).resolve()


def instructions(path: Path) -> list[str]:
    """Dockerfile instructions, with comments dropped and continuations joined."""
    joined: list[str] = []
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        joined.append(" ".join((current + line).split()))
        current = ""
    return joined


def pinned_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class TestVolumes:
    def test_no_named_volumes_anywhere(self, compose: dict) -> None:
        """NEVER named volumes, so copying the project folder carries the data."""
        assert "volumes" not in compose, "top-level volumes: declares named volumes"

    def test_every_mount_is_a_relative_bind(self, services: dict) -> None:
        for name, service in services.items():
            for volume in service.get("volumes", []):
                assert isinstance(volume, str), f"{name}: use the short bind syntax"
                source = volume.split(":", 1)[0]
                assert source.startswith(("./", "../")), f"{name}: {source!r} is not a relative bind mount"

    def test_service_state_lives_under_the_gitignored_data_directory(self, services: dict) -> None:
        for name, service in services.items():
            sources = [v.split(":", 1)[0] for v in service.get("volumes", [])]
            own_state = [s for s in sources if s.startswith("../data/")]
            if name in STATELESS:
                assert not own_state, f"{name}: declared stateless but keeps state in {own_state}"
                continue
            assert own_state, f"{name}: no persistent mount under ../data/"
            assert all(s == f"../data/{name}" for s in own_state), f"{name}: state is not under ../data/{name}"

    def test_every_service_sees_the_data_tree(self, services: dict) -> None:
        for name, service in services.items():
            sources = [v.split(":", 1)[0] for v in service.get("volumes", [])]
            assert "../data" in sources, f"{name}: cannot see data/"

    def test_code_and_configuration_are_mounted_read_only(self, services: dict) -> None:
        """A service may write data and its own declared state, and nothing else."""
        for name, service in services.items():
            for volume in service.get("volumes", []):
                source = volume.split(":", 1)[0]
                if source == "../data" or source.startswith("../data/"):
                    continue
                if source in WRITABLE_OWN_STATE.get(name, set()):
                    continue
                assert volume.endswith(":ro"), f"{name}: {volume!r} should be mounted read-only"

    def test_the_writable_exceptions_are_declared_per_service(self, services: dict) -> None:
        """An allowance names one service and one path; it is not a blanket."""
        for name, sources in WRITABLE_OWN_STATE.items():
            mounted = {v.split(":", 1)[0] for v in services[name].get("volumes", [])}
            assert sources <= mounted, f"{name}: declared a writable path it does not mount"
            for other, service in services.items():
                if other == name:
                    continue
                writable = {v.split(":", 1)[0] for v in service.get("volumes", []) if not v.endswith(":ro")}
                assert not (sources & writable), f"{other} also writes {sources & writable}"


class TestImages:
    def test_every_image_is_tagged_with_an_exact_version(self, services: dict) -> None:
        for name, service in services.items():
            tag = service["image"].rsplit(":", 1)[1]
            assert re.fullmatch(r"\d+(\.\d+)+", tag), f"{name}: {tag!r} is not an exact version"

    def test_every_image_is_built_from_the_official_python_base_pinned_to_a_patch(self, services: dict) -> None:
        for name, service in services.items():
            assert "build" in service, f"{name}: not built here, so its base is not pinned here"
            bases = re.findall(r"^FROM (\S+)", dockerfile_path(service).read_text(encoding="utf-8"), re.MULTILINE)
            assert len(bases) == 1 and BASE_IMAGE.fullmatch(bases[0]), f"{name}: base {bases}"

    def test_every_build_context_has_a_dockerignore_that_keeps_data_and_secrets_out(self, services: dict) -> None:
        for name, service in services.items():
            ignore = (SERVICIOS / service["build"]["context"]).resolve() / ".dockerignore"
            assert ignore.exists(), f"{name}: build context has no .dockerignore"
            rules = ignore.read_text(encoding="utf-8").splitlines()
            assert "data/" in rules and "**/.env" in rules, f"{name}: data or .env can reach a layer"

    def test_jupyter_shares_the_ml_dependency_layer(self, services: dict) -> None:
        """An identical prefix lets BuildKit store the ~3 GB CUDA stack once.
        A one-character drift would silently double it on disk."""
        ml = instructions(dockerfile_path(services["ml"]))
        jupyter = instructions(dockerfile_path(services["jupyter"]))
        end = next(i for i, ins in enumerate(ml) if ins.startswith("RUN pip install") and "/tmp/requirements.txt" in ins)
        assert jupyter[: end + 1] == ml[: end + 1]

    def test_every_image_runs_as_uid_1000(self, services: dict) -> None:
        for name, service in services.items():
            users = [i for i in instructions(dockerfile_path(service)) if i.startswith("USER ")]
            assert users and users[-1] == "USER 1000:1000", f"{name}: runs as {users[-1] if users else 'root'}"


class TestRequirements:
    @pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
    def test_every_requirement_is_pinned_exactly(self, name: str) -> None:
        for line in pinned_lines(name):
            assert PIN.fullmatch(line), f"{name}: {line!r} is not an exact == pin"

    def test_no_package_is_pinned_twice(self) -> None:
        def names(file: str) -> set[str]:
            return {line.split("==")[0].lower() for line in pinned_lines(file)}

        assert not names("requirements.txt") & names("requirements-dev.txt")


class TestProfilesAndNetwork:
    def test_every_service_declares_its_profiles(self, services: dict) -> None:
        assert set(services) == set(PROFILES), "a new service needs a profile in PROFILES"
        for name, service in services.items():
            assert service.get("profiles") == PROFILES[name], f"{name}: wrong profiles"

    def test_one_custom_bridge_network(self, compose: dict) -> None:
        networks = compose["networks"]
        assert len(networks) == 1
        (definition,) = networks.values()
        assert definition["driver"] == "bridge"
        assert "${COMPOSE_PROJECT_NAME}" in definition["name"]

    def test_every_service_joins_it(self, compose: dict, services: dict) -> None:
        network = next(iter(compose["networks"]))
        for name, service in services.items():
            assert service.get("networks") == [network], f"{name}: not on {network}"


class TestRuntimePolicy:
    def test_restart_policies(self, services: dict) -> None:
        """A bare `restart: no` parses as boolean False, so the exception must
        be the quoted string "no" -- this catches the unquoted form too."""
        for name, service in services.items():
            if name in RESTART_EXCEPTIONS:
                assert service.get("restart") == RESTART_EXCEPTIONS[name], f"{name}: declared exception changed"
            else:
                assert service.get("restart") in RESTART_POLICIES, f"{name}: restart policy missing"

    def test_every_long_running_service_has_a_health_check(self, services: dict) -> None:
        for name, service in services.items():
            if name not in RESTART_EXCEPTIONS:
                assert "healthcheck" in service, f"{name}: no health check"

    @pytest.mark.parametrize("name", sorted(LIMITS))
    def test_resource_limits_match_the_global_table(self, services: dict, name: str) -> None:
        limits = services[name]["deploy"]["resources"]["limits"]
        assert limits["memory"] == LIMITS[name]["memory"]
        assert str(limits["cpus"]) == LIMITS[name]["cpus"]

    def test_only_the_training_service_reserves_the_gpu(self, services: dict) -> None:
        for name, service in services.items():
            devices = service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
            if name in GPU_SERVICES:
                assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}], name
            else:
                assert not devices, f"{name}: reserves a GPU that belongs to training"

    def test_the_training_service_has_shared_memory_for_dataloader_workers(self, services: dict) -> None:
        assert services["ml"].get("shm_size") == "1gb"

    def test_host_bind_addresses_are_settings_not_literals(self, services: dict) -> None:
        for name, service in services.items():
            for mapping in service.get("ports", []):
                bind = mapping.partition(":")[0]
                assert VAR_PATTERN.fullmatch(bind), f"{name}: bind address {bind!r} is a literal"
                assert bind.endswith("_BIND_ADDRESS}"), f"{name}: {bind!r} is not a *_BIND_ADDRESS setting"

    def test_services_default_to_localhost(self) -> None:
        addresses = {k: v for k, v in env_values(ENV_EXAMPLE).items() if k.endswith("_BIND_ADDRESS")}
        assert addresses, "no bind address settings in .env.example"
        for key, value in addresses.items():
            assert value == "127.0.0.1", key


class TestCredentials:
    def test_every_compose_variable_is_documented(self, compose: dict) -> None:
        """Scanned over the parsed document: a comment mentioning ${VAR} is prose."""
        used = set(VAR_PATTERN.findall(yaml.safe_dump(compose)))
        missing = used - set(env_values(ENV_EXAMPLE))
        assert not missing, f"used in compose but not in .env.example: {sorted(missing)}"

    def test_example_carries_placeholders_only(self) -> None:
        for key, value in env_values(ENV_EXAMPLE).items():
            if SECRET_KEY.search(key):
                assert value == "CHANGE_ME", f"{key}: real value in .env.example"

    def test_compose_project_name_is_defined(self) -> None:
        assert env_values(ENV_EXAMPLE).get("COMPOSE_PROJECT_NAME")

    def test_secrets_are_injected_from_env(self, services: dict) -> None:
        for name, service in services.items():
            for key, value in service.get("environment", {}).items():
                if SECRET_KEY.search(key):
                    assert VAR_PATTERN.fullmatch(str(value)), f"{name}.{key}: not injected from .env"

    @pytest.mark.skipif(not ENV_FILE.exists(), reason="no local .env (fresh clone)")
    def test_env_and_example_carry_the_same_keys(self) -> None:
        assert set(env_values(ENV_FILE)) == set(env_values(ENV_EXAMPLE))

    @pytest.mark.skipif(not ENV_FILE.exists(), reason="no local .env (fresh clone)")
    def test_local_secrets_do_not_appear_in_any_project_file(self) -> None:
        """Reads .env to look for leaks and never prints what it read."""
        secrets = {v for k, v in env_values(ENV_FILE).items() if SECRET_KEY.search(k) and len(v) > 8}
        assert secrets, ".env has no secrets to check"
        leaked: list[str] = []
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for file in files:
                path = Path(root) / file
                if path == ENV_FILE:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
                if any(secret in content for secret in secrets):
                    leaked.append(str(path.relative_to(PROJECT_ROOT)))
        assert not leaked, f"a value from .env appears in {leaked}"

    def test_gitignore_covers_env_and_data(self) -> None:
        rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "**/.env" in rules
        assert "data/" in rules


class TestPytestConfiguration:
    def test_markers_are_registered_and_the_network_suite_is_opt_in(self) -> None:
        parser = configparser.ConfigParser()
        parser.read(PROJECT_ROOT / "pytest.ini", encoding="utf-8")
        markers = {line.split(":")[0].strip() for line in parser["pytest"]["markers"].splitlines() if line.strip()}
        assert {"gpu", "network"} <= markers
        assert '-m "not network"' in parser["pytest"]["addopts"]
