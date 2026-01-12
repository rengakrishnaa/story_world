class WorldGuard:
    def __init__(self, invariants):
        self.invariants = invariants

    def validate(self, observation):
        violations = []

        for inv in self.invariants:
            if not inv.rule(observation):
                violations.append(inv.name)

        return violations
