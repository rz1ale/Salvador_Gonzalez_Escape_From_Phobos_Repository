"""Read-only entry-interface smoke check. Does not establish vehicle safety/mission success."""
def run(context):
    duration = float(context.parameters.get("duration_s", 2.0))
    if not 0 < duration <= 10:
        raise ValueError("Smoke duration must be 0..10 simulation seconds")
    start = context.clock()
    samples = 0
    while context.clock() - start < duration:
        context.state()
        samples += 1
        context.sleep(.1)
    context.finish(reason="interface_smoke_completed", samples=samples)
