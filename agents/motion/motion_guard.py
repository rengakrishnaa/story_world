def assert_sparse_motion(input_spec: dict):
    motion = input_spec.get("motion")

    if motion is None:
        raise RuntimeError(
            "Motion config missing. Sparse motion is mandatory."
        )

    engine = motion.get("engine")
    if engine != "sparse":
        raise RuntimeError(
            f"Invalid motion engine '{engine}'. "
            f"Only 'sparse' motion is allowed."
        )

    params = motion.get("params", {})
    if not isinstance(params, dict):
        raise RuntimeError("motion.params must be a dict")

    return motion
