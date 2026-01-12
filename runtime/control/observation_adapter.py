# runtime/control/observation_adapter.py

from typing import Dict, Any, List
from runtime.control.contracts import ExecutionObservation, ArtifactSpec
import uuid


class ObservationAdapter:
    """
    Converts raw worker output into a canonical ExecutionObservation.
    """

    @staticmethod
    def from_worker_payload(payload: Dict[str, Any]) -> ExecutionObservation:
        artifacts: List[ArtifactSpec] = []

        for art in payload.get("artifacts", []):
            artifacts.append(
                ArtifactSpec(
                    artifact_id=str(uuid.uuid4()),
                    beat_id=payload["beat_id"],
                    type=art.get("type"),
                    uri=art.get("uri"),
                    metadata=art.get("metadata", {})
                )
            )

        return ExecutionObservation(
            beat_id=payload["beat_id"],
            success=payload.get("success", False),
            error=payload.get("error"),
            metrics=payload.get("metrics", {}),
            artifacts=artifacts
        )
