class InteractionValidator:
    def validate(self, interactions, entity_scores):
        violations = []

        for inter in interactions:
            subj = inter["subject"]
            obj = inter["object"]

            if entity_scores.get(subj, 0) < 0.3:
                violations.append(f"{subj}_missing")

            if entity_scores.get(obj, 0) < 0.3:
                violations.append(f"{obj}_missing")

        return violations
