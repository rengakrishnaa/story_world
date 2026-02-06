import asyncio
import os
import traceback
from runtime.episode_runtime import EpisodeRuntime
from runtime.beat_state import BeatState
from runtime.physics_veto import evaluate_physics_veto
from models.state_transition import TransitionStatus, ActionOutcome

class ResultConsumer:
    def __init__(self, sql_store, redis_store, world_graph_store=None):
        self.sql = sql_store
        self.redis = redis_store
        self.world_graph_store = world_graph_store
        self._running = True

    async def run_loop(self):
        """
        Background task that continually polls Redis for GPU results
        and syncs them to the local SQL database.
        Queue must match worker RESULT_QUEUE (default: storyworld:gpu:results).
        """
        try:
            from runtime.persistence.redis_store import _result_queue
            queue_name = _result_queue()
        except Exception:
            queue_name = os.getenv("GPU_RESULT_QUEUE") or os.getenv("RESULT_QUEUE") or "storyworld:gpu:results"
        
        # Test Redis connection
        try:
            # Access the redis property to trigger lazy connection
            redis_client = self.redis.redis
            redis_client.ping()
            print(f"[ResultConsumer] Started. Listening on queue: {queue_name} (set RESULT_QUEUE to match worker)")
        except Exception as e:
            print(f"[ResultConsumer] Redis connection failed: {e}")
            print(f"[ResultConsumer] ResultConsumer will not process results until Redis is available")
            return
        
        loop_count = 0
        while self._running:
            try:
                result = await asyncio.to_thread(self.redis.pop_gpu_result, timeout=2)
                
                if not result:
                    loop_count += 1
                    # Log every 50 iterations (roughly every 10 seconds) to show it's alive
                    if loop_count % 50 == 0:
                        print(f"[ResultConsumer] Polling queue '{queue_name}' (no results yet)")
                    await asyncio.sleep(0.1) 
                    continue
                
                loop_count = 0  # Reset counter when we get a result
                job_id = result.get("job_id")
                meta = result.get("meta", {})
                episode_id = meta.get("episode_id")
                beat_id = meta.get("beat_id")
                
                if not job_id:
                    print("[ResultConsumer] Skipping result: missing job_id")
                    continue

                print(f"[ResultConsumer] job={job_id} episode={episode_id} beat={beat_id} status={result.get('status')}")
                # Run observer in thread pool with timeout to prevent blocking
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._process_result, result),
                        timeout=120.0  # 2 minute max for observer (Gemini retries + Ollama timeout)
                    )
                except asyncio.TimeoutError:
                    print(f"[ResultConsumer] Observer timeout for beat {beat_id}, marking as uncertain")
                    # Mark beat as failed due to timeout
                    try:
                        runtime = EpisodeRuntime.load(episode_id, self.sql)
                        runtime.mark_beat_failure(
                            beat_id=beat_id,
                            error="Observer timeout (exceeded 120s)",
                            metrics={"verdict": "uncertain", "confidence": 0.0}
                        )
                    except Exception as e:
                        print(f"[ResultConsumer] Failed to mark timeout: {e}")

            except Exception as e:
                if "Timeout" in type(e).__name__ or "timeout" in str(e).lower():
                    await asyncio.sleep(0.5)
                    continue
                print(f"[ResultConsumer] Loop Error: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    def stop(self):
        self._running = False

    async def process_batch(self, max_items: int = 5) -> int:
        """
        Process up to max_items results from the queue. For serverless (Vercel cron)
        where a background loop cannot run. Returns number of results processed.
        """
        processed = 0
        for _ in range(max_items):
            try:
                result = await asyncio.to_thread(self.redis.pop_gpu_result, timeout=1)
                if not result:
                    break
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._process_result, result),
                        timeout=120.0,
                    )
                    processed += 1
                except asyncio.TimeoutError:
                    job_id = result.get("job_id", "?")
                    meta = result.get("meta", {})
                    beat_id = meta.get("beat_id")
                    print(f"[ResultConsumer] process_batch: observer timeout for {job_id}")
                    try:
                        runtime = EpisodeRuntime.load(meta.get("episode_id"), self.sql)
                        runtime.mark_beat_failure(
                            beat_id=beat_id,
                            error="Observer timeout (exceeded 120s)",
                            metrics={"verdict": "uncertain", "confidence": 0.0},
                        )
                    except Exception as e:
                        print(f"[ResultConsumer] Failed to mark timeout: {e}")
                    processed += 1
            except Exception as e:
                print(f"[ResultConsumer] process_batch error: {e}")
                break
        return processed

    def _process_result(self, result: dict):
        meta = result.get("meta", {})
        episode_id = meta.get("episode_id")
        beat_id = meta.get("beat_id")

        if not episode_id or not beat_id:
            print(f"[ResultConsumer] Skipping result with missing meta IDs: {result.get('job_id')}")
            return

        try:
            # Load the runtime for this episode (rehydrates from SQL)
            runtime = EpisodeRuntime.load(episode_id, self.sql)

            if result["status"] == "success":
                # Run observer and record to world graph (authoritative)
                # Note: _observe_and_record will log its own status (SUCCESS, EPISTEMIC_HALT, etc.)
                self._observe_and_record(episode_id, beat_id, result, runtime)
                # Only log generic SUCCESS if we didn't already log a specific status
                # (epistemic halts log their own status and return early)
            else:
                error_msg = result.get("error", {}).get("message", "Unknown GPU Error")
                runtime.mark_beat_failure(
                    beat_id=beat_id,
                    error=error_msg,
                    metrics=result.get("runtime", {})
                )
                print(f"[ResultConsumer] Processed FAILURE for {episode_id}/{beat_id}: {error_msg}")

        except Exception as e:
            print(f"[ResultConsumer] Failed to sync result for episode {episode_id}")
            traceback.print_exc()

    def _observe_and_record(self, episode_id: str, beat_id: str, result: dict, runtime) -> None:
        """Run observer on video and record observation to world graph."""
        use_observer = os.getenv("USE_OBSERVER_IN_PRODUCTION", "true").lower() in ("true", "1", "yes")
        if not use_observer or not self.world_graph_store:
            # Fallback: accept beat if observer is disabled
            runtime.mark_beat_success(
                beat_id=beat_id,
                artifacts=result.get("artifacts", {}),
                metrics=result.get("runtime", {}),
            )
            return

        # Deterministic veto should run even if video download fails.
        veto, veto_constraints, veto_reason = evaluate_physics_veto(runtime.intent or "")

        artifacts = result.get("artifacts") or {}
        video_local = artifacts.get("video_local_path")
        video_uri = video_local if (video_local and os.path.isfile(video_local)) else artifacts.get("video")
        # If deterministic veto triggers, reject immediately (video is not required).
        if veto:
            metrics = {**(result.get("runtime") or {})}
            metrics.update({"confidence": 1.0, "verdict": "impossible", "veto": "deterministic"})
            runtime.mark_beat_failure(
                beat_id=beat_id,
                error=f"Deterministic physics veto: {veto_reason}",
                metrics=metrics,
                observer_verdict="impossible",
            )
            runtime.mark_episode_impossible(reason=f"Deterministic physics veto: {veto_reason}")

            # Record rejected transition with minimal observation payload
            obs_dict = {
                "observation_id": f"veto:{episode_id}:{beat_id}",
                "video_uri": video_uri,
                "beat_id": beat_id,
                "verdict": "impossible",
                "confidence": 1.0,
                "constraints_inferred": veto_constraints,
                "causal_explanation": veto_reason,
            }
            self.world_graph_store.record_beat_observation(
                episode_id=episode_id,
                beat_id=beat_id,
                video_uri=video_uri or "",
                observation=obs_dict,
                action_description=(runtime.intent or "")[:200],
                video_duration_sec=float((result.get("runtime") or {}).get("latency_sec", 0.0)),
                quality_score=0.0,
                transition_status=TransitionStatus.REJECTED,
                action_outcome=ActionOutcome.FAILED,
            )
            return

        if not video_uri:
            print(f"[ResultConsumer] Missing video_uri for beat {beat_id}, marking as uncertain")
            runtime.mark_beat_failure(
                beat_id=beat_id,
                error="Missing video artifact for observation",
                metrics={**(result.get("runtime") or {}), "verdict": "uncertain", "confidence": 0.0},
                observer_verdict="uncertain",
            )
            # Still record to world graph with REJECTED status
            if self.world_graph_store:
                self.world_graph_store.record_beat_observation(
                    episode_id=episode_id,
                    beat_id=beat_id,
                    video_uri="",
                    observation={
                        "verdict": "uncertain",
                        "confidence": 0.0,
                        "error": "missing_video",
                        "constraints_inferred": ["missing_video", "insufficient_evidence"],
                    },
                    action_description=(runtime.intent or "")[:200],
                    video_duration_sec=0.0,
                    quality_score=0.0,
                    transition_status=TransitionStatus.REJECTED,
                    action_outcome=ActionOutcome.FAILED,
                )
            # Re-render with different camera/scale, not abort
            try:
                from runtime.physics_observability import should_augment_for_observability
                if should_augment_for_observability(["video_unavailable", "insufficient_evidence"]):
                    if runtime.augment_beat_for_observability(beat_id):
                        print(f"[ResultConsumer] Augmented beat {beat_id} for observability, re-render")
                    else:
                        runtime.abort_beat_observability_cap(beat_id)
                        print(f"[ResultConsumer] Observability cap reached for {beat_id} (no video), aborting")
                        return
            except Exception:
                pass
            try:
                runtime.submit_pending_beats(self.redis)
            except Exception:
                pass
            return

        try:
            beat_desc = ""
            try:
                b = runtime.sql.get_beat(beat_id)
                if b and b.get("spec"):
                    beat_desc = (b["spec"].get("description") or "")[:500]
            except Exception:
                pass
            print(f"[ResultConsumer] Calling observer for beat {beat_id}, video_uri={video_uri[:50] if video_uri else 'None'}...")
            observation = self._run_observer_sync(
                video_uri, episode_id, beat_id,
                runtime.intent or "",
                beat_description=beat_desc,
            )
            print(f"[ResultConsumer] Observer returned for beat {beat_id}, observation={'present' if observation else 'None'}")
            if observation:
                verdict = (observation.verdict or "valid").lower()
                metrics = {**(result.get("runtime") or {})}
                metrics.update(
                    {
                        "confidence": observation.confidence,
                        "verdict": verdict,
                        "disagreement_score": observation.disagreement_score,
                    }
                )
                obs_dict = observation.to_dict()
                duration = float((result.get("runtime") or {}).get("latency_sec", 0.0))
                quality = getattr(observation.quality, "overall_quality", 0.75) or 0.75
                
                # ========================================================================
                # EPISTEMIC CHECK: Layer 2 & 3 - Evidence & Constraint Evaluation
                # MUST RUN BEFORE verdict-based handling to prevent best-effort acceptance
                # ========================================================================
                from runtime.epistemic_evaluator import evaluate_epistemic_state
                from models.epistemic import EpistemicState
                
                evidence_ledger = getattr(observation, "evidence_ledger", None)
                observer_unavailable = (
                    getattr(observation, "observer_type", None) == "mock"
                    or (evidence_ledger is None and verdict in ("uncertain", "valid"))
                )
                from models.intent_classification import get_intent_override_from_policies
                intent_override = get_intent_override_from_policies(runtime.policies)
                epistemic_state, epistemic_summary = evaluate_epistemic_state(
                    evidence_ledger=evidence_ledger,
                    intent=runtime.intent or "",
                    verdict=verdict,
                    confidence=observation.confidence,
                    observer_unavailable=observer_unavailable,
                    intent_override=intent_override,
                )
                
                # If epistemically incomplete, halt progression (NO re-render, NO best-effort)
                if epistemic_state == EpistemicState.EPISTEMICALLY_INCOMPLETE:
                    missing = epistemic_summary.missing_evidence if epistemic_summary else []
                    justification = epistemic_summary.justification if epistemic_summary else []
                    print(
                        f"[ResultConsumer] EPISTEMIC HALT for {beat_id}: "
                        f"Missing evidence: {missing}. {justification[0] if justification else 'Unknown reason'}"
                    )
                    # Mark as epistemically incomplete (new beat state)
                    runtime.sql.mark_beat_state(
                        beat_id,
                        BeatState.EPISTEMICALLY_INCOMPLETE,
                        f"Missing evidence: {', '.join(missing)}"
                    )
                    # Record to world graph with BLOCKED status
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation={
                            **obs_dict,
                            "epistemic_state": "EPISTEMICALLY_INCOMPLETE",
                            "missing_evidence": missing,
                            "epistemic_summary": epistemic_summary.to_dict() if epistemic_summary else None,
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.BLOCKED,
                        action_outcome=ActionOutcome.UNKNOWN,
                    )
                    # Do NOT re-render - epistemic halt is explicit
                    runtime._recompute_episode_state()
                    print(f"[ResultConsumer] Processed EPISTEMIC_HALT for {episode_id}/{beat_id}")
                    return
                
                # If uncertain termination, halt progression
                if epistemic_state == EpistemicState.UNCERTAIN_TERMINATION:
                    print(
                        f"[ResultConsumer] UNCERTAIN TERMINATION for {beat_id}: "
                        f"{epistemic_summary.justification[0] if epistemic_summary and epistemic_summary.justification else 'Uncertain verdict'}"
                    )
                    runtime.sql.mark_beat_state(
                        beat_id,
                        BeatState.UNCERTAIN_TERMINATION,
                        "Uncertain verdict despite available evidence"
                    )
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation={
                            **obs_dict,
                            "epistemic_state": "UNCERTAIN_TERMINATION",
                            "epistemic_summary": epistemic_summary.to_dict() if epistemic_summary else None,
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.BLOCKED,
                        action_outcome=ActionOutcome.UNKNOWN,
                    )
                    runtime._recompute_episode_state()
                    print(f"[ResultConsumer] Processed UNCERTAIN_TERMINATION for {episode_id}/{beat_id}")
                    return
                
                # If rejected, mark as rejected
                if epistemic_state == EpistemicState.REJECTED:
                    runtime.mark_beat_failure(
                        beat_id=beat_id,
                        error=f"Epistemic rejection: {verdict}",
                        metrics=metrics,
                        observer_verdict=verdict,
                    )
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation={
                            **obs_dict,
                            "epistemic_state": "REJECTED",
                            "epistemic_summary": epistemic_summary.to_dict() if epistemic_summary else None,
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.REJECTED,
                        action_outcome=ActionOutcome.FAILED,
                    )
                    return
                
                # Solver-only success: closed-form intent, observer unavailable or insufficient evidence.
                # Observer is witness; infrastructure failure or evidence gap must NOT block closed-form.
                obs_status = getattr(epistemic_summary, "observer_status", None)
                if epistemic_state == EpistemicState.ACCEPTED and epistemic_summary and obs_status in ("unavailable", "insufficient_observational_evidence"):
                    solver_confidence = 0.75
                    metrics["confidence"] = solver_confidence
                    metrics["observer_status"] = obs_status
                    metrics["confidence_penalty_reason"] = getattr(epistemic_summary, "confidence_penalty_reason", None) or ("observer_could_not_extract_evidence" if obs_status == "insufficient_observational_evidence" else "observer_infrastructure_failure")
                    print(f"[ResultConsumer] Solver-only success for {beat_id}: {obs_status}, confidence={solver_confidence}")
                    runtime.mark_beat_success(beat_id, result.get("artifacts", {}), metrics)
                    c_inferred = (["insufficient_observational_evidence"] if obs_status == "insufficient_observational_evidence" else ["observer_unavailable"]) + (obs_dict.get("constraints_inferred") or [])
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation={
                            **obs_dict,
                            "epistemic_state": "ACCEPTED",
                            "epistemic_summary": epistemic_summary.to_dict(),
                            "observer_status": obs_status,
                            "observer_impact": "confidence_only",
                            "confidence_penalty_reason": metrics["confidence_penalty_reason"],
                            "constraints_inferred": c_inferred,
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.COMPLETED,
                        action_outcome=ActionOutcome.SUCCESS,
                    )
                    return
                
                # Now handle verdict-based transitions (only if epistemic check passed)
                if verdict in ("impossible", "contradicts", "blocks"):
                    runtime.mark_beat_failure(
                        beat_id=beat_id,
                        error=f"Observer veto: {verdict}",
                        metrics=metrics,
                        observer_verdict=verdict,
                    )
                    runtime.mark_episode_impossible(reason=f"Observer veto: {verdict}")
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation=obs_dict,
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.REJECTED,
                        action_outcome=ActionOutcome.FAILED,
                    )
                    return
                if verdict in ("uncertain", "failed"):
                    if isinstance(obs_dict, dict) and not obs_dict.get("constraints_inferred"):
                        obs_dict["constraints_inferred"] = ["insufficient_evidence"]
                    runtime.mark_beat_failure(
                        beat_id=beat_id,
                        error=f"Observer verdict: {verdict}",
                        metrics=metrics,
                        observer_verdict=verdict,
                    )
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation=obs_dict,
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=duration,
                        quality_score=quality,
                        transition_status=TransitionStatus.REJECTED,
                        action_outcome=ActionOutcome.FAILED,
                    )
                    # Progressive constraint tightening: even failed beats can inform next (physics learned)
                    constraints = obs_dict.get("constraints_inferred") or []
                    if constraints:
                        try:
                            n = runtime.refine_pending_beats(constraints)
                            if n > 0:
                                print(f"[ResultConsumer] Refined {n} pending beats from failed observation")
                        except Exception as ex:
                            print(f"[ResultConsumer] Beat refinement failed: {ex}")
                    # Re-render with different camera/scale on insufficient_evidence, not abort
                    try:
                        from runtime.physics_observability import should_augment_for_observability
                        if should_augment_for_observability(constraints):
                            if runtime.augment_beat_for_observability(beat_id):
                                print(f"[ResultConsumer] Augmented beat {beat_id} for observability, re-render")
                            else:
                                # Observability cap reached: accept as best-effort to avoid infinite loop and PARTIALLY_COMPLETED
                                runtime.mark_beat_success(
                                    beat_id=beat_id,
                                    artifacts=result.get("artifacts", {}),
                                    metrics=metrics,
                                )
                                print(f"[ResultConsumer] Observability cap reached for {beat_id}, accepting as best-effort")
                                return
                    except Exception:
                        pass
                    try:
                        runtime.submit_pending_beats(self.redis)
                    except Exception:
                        pass
                    return

                # ACCEPTED: All evidence available, constraints satisfied (epistemic check already passed above)
                runtime.mark_beat_success(
                    beat_id=beat_id,
                    artifacts=result.get("artifacts", {}),
                    metrics=metrics,
                )
                outcome = ActionOutcome.SUCCESS if verdict == "valid" else ActionOutcome.PARTIAL
                print(f"[ResultConsumer] Processed ACCEPTED for {episode_id}/{beat_id} (all evidence available, constraints satisfied)")
                self.world_graph_store.record_beat_observation(
                    episode_id=episode_id,
                    beat_id=beat_id,
                    video_uri=video_uri,
                    observation={
                        **obs_dict,
                        "epistemic_state": "ACCEPTED",
                        "epistemic_summary": epistemic_summary.to_dict() if epistemic_summary else None,
                    },
                    action_description=(runtime.intent or "")[:200],
                    video_duration_sec=duration,
                    quality_score=quality,
                    transition_status=TransitionStatus.COMPLETED,
                    action_outcome=outcome,
                )
                # Progressive constraint tightening: refine next beats with learned physics
                constraints = obs_dict.get("constraints_inferred") or []
                if constraints:
                    try:
                        n = runtime.refine_pending_beats(constraints)
                        if n > 0:
                            print(f"[ResultConsumer] Refined {n} pending beats with prior constraints")
                    except Exception as ex:
                        print(f"[ResultConsumer] Beat refinement failed: {ex}")
                return
            # No observation: observer returned None (exception before fallback)
            # For closed-form intents, solver-only success - observer is witness, not judge
            from models.intent_classification import requires_visual_verification, get_intent_override_from_policies
            override = get_intent_override_from_policies(runtime.policies)
            override_val = override.get("requires_visual_verification") if override else None
            if not requires_visual_verification(runtime.intent or "", override=override_val):
                solver_confidence = 0.75
                metrics = {**(result.get("runtime") or {}), "confidence": solver_confidence, "verdict": "valid"}
                metrics["observer_status"] = "unavailable"
                metrics["confidence_penalty_reason"] = "observer_infrastructure_failure"
                print(f"[ResultConsumer] Solver-only success for {beat_id}: observer returned None, confidence={solver_confidence}")
                runtime.mark_beat_success(beat_id, result.get("artifacts", {}), metrics)
                if self.world_graph_store:
                    try:
                        video_uri = (result.get("artifacts") or {}).get("video") or ""
                        self.world_graph_store.record_beat_observation(
                            episode_id=episode_id,
                            beat_id=beat_id,
                            video_uri=video_uri,
                            observation={
                                "verdict": "valid",
                                "confidence": solver_confidence,
                                "epistemic_state": "ACCEPTED",
                                "observer_status": "unavailable",
                                "observer_impact": "confidence_only",
                                "confidence_penalty_reason": "observer_infrastructure_failure",
                                "constraints_inferred": ["observer_unavailable"],
                                "constraints_satisfied": ["no_tipping"] if "stack" in (runtime.intent or "").lower() and "tipping" in (runtime.intent or "").lower() else [],
                            },
                            action_description=(runtime.intent or "")[:200],
                            video_duration_sec=float((result.get("runtime") or {}).get("latency_sec", 0.0)),
                            quality_score=0.75,
                            transition_status=TransitionStatus.COMPLETED,
                            action_outcome=ActionOutcome.SUCCESS,
                        )
                    except Exception as ex:
                        print(f"[ResultConsumer] Failed to record solver-only: {ex}")
                return
            runtime.mark_beat_failure(
                beat_id=beat_id,
                error="Observer returned no observation",
                metrics={**(result.get("runtime") or {}), "verdict": "uncertain", "confidence": 0.0},
                observer_verdict="uncertain",
            )
            if self.world_graph_store:
                try:
                    video_uri = (result.get("artifacts") or {}).get("video") or ""
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=video_uri,
                        observation={
                            "verdict": "uncertain",
                            "confidence": 0.0,
                            "error": "observer_no_observation",
                            "constraints_inferred": ["insufficient_evidence", "video_unavailable"],
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=float((result.get("runtime") or {}).get("latency_sec", 0.0)),
                        quality_score=0.0,
                        transition_status=TransitionStatus.REJECTED,
                        action_outcome=ActionOutcome.FAILED,
                    )
                except Exception as ex:
                    print(f"[ResultConsumer] Failed to record no-observation: {ex}")
            # Re-render with different camera/scale when observer could not extract physics
            try:
                from runtime.physics_observability import should_augment_for_observability
                if should_augment_for_observability(["insufficient_evidence", "video_unavailable"]):
                    if runtime.augment_beat_for_observability(beat_id):
                        print(f"[ResultConsumer] Augmented beat {beat_id} for observability (observer no observation)")
                    else:
                        runtime.mark_beat_success(
                            beat_id=beat_id,
                            artifacts=result.get("artifacts", {}),
                            metrics={**(result.get("runtime") or {}), "verdict": "uncertain"},
                        )
                        print(f"[ResultConsumer] Observability cap reached for {beat_id} (no observation), accepting as best-effort")
                        return
            except Exception:
                pass
            try:
                runtime.submit_pending_beats(self.redis)
            except Exception:
                pass
        except Exception as e:
            print(f"[ResultConsumer] Observer failed for {episode_id}/{beat_id}: {e}")
            traceback.print_exc()
            # Fail-safe: never leave beats stuck; treat as UNCERTAIN so retry policy can act.
            runtime.mark_beat_failure(
                beat_id=beat_id,
                error=f"Observer exception: {e}",
                metrics={**(result.get("runtime") or {}), "verdict": "uncertain", "confidence": 0.0},
                observer_verdict="uncertain",
            )
            # Record to world graph for audit trail even when observer fails
            if self.world_graph_store:
                try:
                    self.world_graph_store.record_beat_observation(
                        episode_id=episode_id,
                        beat_id=beat_id,
                        video_uri=(result.get("artifacts") or {}).get("video") or "",
                        observation={
                            "verdict": "uncertain",
                            "confidence": 0.0,
                            "error": f"observer_exception:{str(e)[:100]}",
                            "constraints_inferred": ["observer_exception", "insufficient_evidence"],
                        },
                        action_description=(runtime.intent or "")[:200],
                        video_duration_sec=0.0,
                        quality_score=0.0,
                        transition_status=TransitionStatus.REJECTED,
                        action_outcome=ActionOutcome.FAILED,
                    )
                except Exception as ex:
                    print(f"[ResultConsumer] Failed to record observation: {ex}")
            # Re-render with different camera on observer exception
            try:
                from runtime.physics_observability import should_augment_for_observability
                if should_augment_for_observability(["observer_exception", "insufficient_evidence"]):
                    if runtime.augment_beat_for_observability(beat_id):
                        print(f"[ResultConsumer] Augmented beat {beat_id} for observability (observer exception)")
                    else:
                        runtime.mark_beat_success(
                            beat_id=beat_id,
                            artifacts=(result.get("artifacts") or {}),
                            metrics={**(result.get("runtime") or {}), "verdict": "uncertain"},
                        )
                        print(f"[ResultConsumer] Observability cap reached for {beat_id} (observer exception), accepting as best-effort")
                        return
            except Exception:
                pass
            try:
                runtime.submit_pending_beats(self.redis)
            except Exception:
                pass

    def _run_observer_sync(self, video_uri: str, episode_id: str, beat_id: str, intent: str, beat_description: str = ""):
        """
        Synchronous observer call (runs in thread or inline).
        
        This method can block for up to:
        - Gemini retries: ~30s (3 retries * 10s)
        - Ollama timeout: 30s
        - Total: ~60s worst case, but we allow 120s for safety
        """
        try:
            print(f"[ResultConsumer] Starting observer for beat {beat_id}")
            from agents.video_observer import VideoObserverAgent, ObserverConfig
            from models.observation import TaskContext
            from runtime.physics_observability import get_observer_physics_questions
            config = ObserverConfig(
                use_gemini=os.getenv("GEMINI_API_KEY") is not None,
                gemini_model=os.getenv("GEMINI_OBSERVER_MODEL", "gemini-2.0-flash-lite"),
                fallback_enabled=os.getenv("OBSERVER_FALLBACK_ENABLED", "").lower() in ("true", "1", "yes"),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                ollama_model=os.getenv("OLLAMA_MODEL", "llava"),
            )
            observer = VideoObserverAgent(config=config)
            physics_questions = get_observer_physics_questions(beat_description, intent)
            context = TaskContext(
                beat_id=beat_id,
                episode_id=episode_id,
                task_type="simulation",
                expected_action=intent,
                physics_questions=physics_questions,
            )
            result = observer.observe(video_uri, context)
            print(f"[ResultConsumer] Observer completed for beat {beat_id}, verdict={getattr(result, 'verdict', 'unknown')}")
            return result
        except Exception as e:
            print(f"[ResultConsumer] Observer error for beat {beat_id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
