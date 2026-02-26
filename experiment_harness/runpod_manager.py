"""RunPod API wrapper: create and destroy GPU pods."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

RUNPOD_API_URL = "https://api.runpod.io/graphql"


@dataclass
class PodInfo:
    """Info about a running RunPod pod."""
    pod_id: str
    status: str
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    gpu_type: Optional[str] = None
    cost_per_hr: Optional[float] = None


class RunPodManager:
    """Create/destroy RunPod instances via the GraphQL API."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _query(self, query: str, variables: Optional[dict] = None) -> dict:
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = requests.post(RUNPOD_API_URL, json=payload, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data["data"]

    def create_pod(
        self,
        name: str = "experiment-harness",
        gpu_type: str = "NVIDIA RTX A5000",
        cloud_type: str = "COMMUNITY",
        docker_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        volume_size_gb: int = 50,
        container_disk_gb: int = 20,
    ) -> PodInfo:
        """Create a new GPU pod and wait for it to be ready."""
        print(f"[RunPod] Creating pod: {name} with {gpu_type}...")

        query = """
        mutation {{
            podFindAndDeployOnDemand(input: {{
                name: "{name}"
                gpuTypeId: "{gpu_type}"
                cloudType: {cloud_type}
                imageName: "{docker_image}"
                volumeInGb: {volume_size_gb}
                containerDiskInGb: {container_disk_gb}
                startSsh: true
                supportPublicIp: true
            }}) {{
                id
                desiredStatus
                imageName
                machineId
                costPerHr
                gpuCount
            }}
        }}
        """.format(
            name=name,
            gpu_type=gpu_type,
            cloud_type=cloud_type,
            docker_image=docker_image,
            volume_size_gb=volume_size_gb,
            container_disk_gb=container_disk_gb,
        )

        data = self._query(query)
        pod_data = data["podFindAndDeployOnDemand"]
        pod_id = pod_data["id"]
        cost = pod_data.get("costPerHr")

        print(f"[RunPod] Pod created: {pod_id} (${cost}/hr)")
        print(f"[RunPod] Waiting for pod to be ready...")

        # Poll until running and SSH is available
        pod_info = self._wait_for_ready(pod_id, timeout=300)
        pod_info.cost_per_hr = cost
        return pod_info

    def _wait_for_ready(self, pod_id: str, timeout: int = 300) -> PodInfo:
        """Poll until pod is RUNNING and has SSH info."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.get_pod(pod_id)
            if info.status == "RUNNING" and info.ssh_host:
                print(f"[RunPod] Pod ready: {info.ssh_host}:{info.ssh_port}")
                return info
            print(f"[RunPod] Status: {info.status}...")
            time.sleep(10)

        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")

    def get_pod(self, pod_id: str) -> PodInfo:
        """Get current info about a pod."""
        query = """
        query {{
            pod(input: {{ podId: "{pod_id}" }}) {{
                id
                desiredStatus
                runtime {{
                    uptimeInSeconds
                    ports {{
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }}
                    gpus {{
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }}
                }}
                gpuCount
                costPerHr
                machine {{
                    gpuDisplayName
                }}
            }}
        }}
        """.format(pod_id=pod_id)

        data = self._query(query)
        pod = data["pod"]

        ssh_host = None
        ssh_port = None
        if pod.get("runtime") and pod["runtime"].get("ports"):
            for port_info in pod["runtime"]["ports"]:
                if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                    ssh_host = port_info["ip"]
                    ssh_port = port_info["publicPort"]
                    break

        return PodInfo(
            pod_id=pod["id"],
            status=pod["desiredStatus"],
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
            cost_per_hr=pod.get("costPerHr"),
        )

    def stop_pod(self, pod_id: str) -> None:
        """Stop a running pod."""
        print(f"[RunPod] Stopping pod {pod_id}...")
        query = """
        mutation {{
            podStop(input: {{ podId: "{pod_id}" }}) {{
                id
                desiredStatus
            }}
        }}
        """.format(pod_id=pod_id)
        self._query(query)
        print(f"[RunPod] Pod stopped.")

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate and delete a pod."""
        print(f"[RunPod] Terminating pod {pod_id}...")
        query = """
        mutation {{
            podTerminate(input: {{ podId: "{pod_id}" }})
        }}
        """.format(pod_id=pod_id)
        self._query(query)
        print(f"[RunPod] Pod terminated.")

    def list_gpus(self) -> list[dict]:
        """List available GPU types."""
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                communityPrice
                securePrice
            }
        }
        """
        data = self._query(query)
        return data["gpuTypes"]
