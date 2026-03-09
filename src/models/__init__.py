def make_agent(envs, attention="none"):
    if attention == "none":
        from .plain import Agent
    elif attention == "single":
        from .single_head import Agent
    elif attention == "multi":
        from .multi_head import Agent
    else:
        raise ValueError(f"Unknown attention type: {attention!r}")
    return Agent(envs)
