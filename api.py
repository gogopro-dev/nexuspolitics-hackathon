from model import classify_issue


def run_model(description: str, category: str, state: str) -> str:
    authority = classify_issue(description, category, state)
    return f"Processed: ({authority.id}] {[authority.level]}]) {authority.name}"
