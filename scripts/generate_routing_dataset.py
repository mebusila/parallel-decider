from __future__ import annotations

import json
import random
from pathlib import Path

from parallel_decider.dataset import ROUTING_LABELS
from parallel_decider.dataset import load_routing_dataset

SEED = 42

OUTPUT_PATH = Path("data/routing_v3_candidates.jsonl")

ACTION_VERBS = {
    "read": [
        "Read",
        "Inspect",
        "Open",
        "Check",
    ],
    "write": [
        "Save",
        "Update",
        "Modify",
        "Write",
    ],
    "execute": [
        "Run",
        "Execute",
        "Launch",
    ],
    "explain": [
        "Explain",
        "Describe",
        "Summarize",
    ],
}


PARAPHRASE_PREFIXES = [
    "",
    "Please ",
    "I need you to ",
    "Can you ",
]

PARAPHRASE_SUFFIXES = [
    "",
    " and report the result.",
    " and give me a concise summary.",
]

EXPANDED_OUTPUT_PATH = Path("data/routing_v3_expanded.jsonl")

TRAINING_OUTPUT_PATH = Path("data/routing_v3_training.jsonl")


def normalize_sentence(text: str) -> str:
    text = text.strip()

    if not text:
        return text

    return text[0].lower() + text[1:]


def paraphrase_variants(
    item: dict,
    max_variants: int = 3,
) -> list[dict]:
    original = item["state"].strip()

    variants = [
        item,
    ]

    lower_original = normalize_sentence(original)

    candidates = []

    for prefix in PARAPHRASE_PREFIXES:
        if not prefix:
            continue

        candidates.append(
            {
                "state": prefix + lower_original,
                "labels": dict(item["labels"]),
            }
        )

    for suffix in PARAPHRASE_SUFFIXES:
        if not suffix:
            continue

        if original.endswith("."):
            base = original[:-1]
        else:
            base = original

        candidates.append(
            {
                "state": base + suffix,
                "labels": dict(item["labels"]),
            }
        )

    for candidate in candidates:
        if len(variants) >= max_variants:
            break

        if candidate["state"] != original:
            variants.append(candidate)

    return variants


def labels(**enabled: int) -> dict[str, int]:
    result = {label: 0 for label in ROUTING_LABELS}

    for label, value in enabled.items():
        if label not in result:
            raise ValueError(f"Unknown routing label: {label}")

        result[label] = int(value)

    return result


def example(
    state: str,
    **enabled: int,
) -> dict:
    return {
        "state": state,
        "labels": labels(**enabled),
    }


def write_jsonl(
    path: Path,
    examples: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def deduplicate(
    examples: list[dict],
) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []

    for item in examples:
        state = item["state"].strip()

        if state in seen:
            continue

        seen.add(state)
        unique.append(item)

    return unique


def print_coverage(examples: list[dict]) -> None:
    total = len(examples)

    print(f"\nTotal examples: {total}")

    print("\nPositive examples per capability:")

    for label in ROUTING_LABELS:
        positives = sum(item["labels"][label] for item in examples)

        print(f"{label:<20} " f"{positives:>3} " f"({positives / total:>6.1%})")

    counts_by_active_labels: dict[int, int] = {}

    for item in examples:
        active = sum(item["labels"].values())

        counts_by_active_labels[active] = counts_by_active_labels.get(active, 0) + 1

    print("\nExamples by number of active capabilities:")

    for active_count in sorted(counts_by_active_labels):
        count = counts_by_active_labels[active_count]

        print(
            f"{active_count} capabilities: " f"{count:>3} " f"({count / total:>6.1%})"
        )


def filesystem_examples() -> list[dict]:
    return [
        example(
            "Read the local application settings file and summarize the configured limits.",
            needs_filesystem=1,
        ),
        example(
            "Update the local configuration file so debug logging is disabled.",
            needs_filesystem=1,
        ),
        example(
            "Explain what a configuration file is without opening any local files.",
        ),
        example(
            "Save the generated summary into a local Markdown file.",
            needs_filesystem=1,
        ),
    ]


def shell_examples() -> list[dict]:
    return [
        example(
            "Run the project's unit tests and report the failing test names.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Show me an example pytest command, but do not execute it.",
        ),
        example(
            "Execute the local benchmark script and report the runtime.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Explain what the uname command does.",
        ),
    ]


def git_examples() -> list[dict]:
    return [
        example(
            "Find the commit that introduced the current parser implementation.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Create a local Git branch named experiment/cache-v2.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Explain the difference between a Git tag and a branch.",
        ),
        example(
            "Compare the current working tree with the previous commit.",
            needs_filesystem=1,
            needs_git=1,
        ),
    ]


def browser_network_examples() -> list[dict]:
    return [
        example(
            "Check the official Python documentation for the current typing syntax.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Explain Python type annotations from your existing knowledge.",
        ),
        example(
            "Check whether the supplied API endpoint is reachable.",
            needs_network=1,
        ),
        example(
            "Search the vendor website for the latest API documentation.",
            needs_browser=1,
            needs_network=1,
        ),
    ]


def database_examples() -> list[dict]:
    return [
        example(
            "Find all accounts created today in the customer database.",
            needs_database=1,
        ),
        example(
            "Update the database record for order 8821 to mark it as shipped.",
            needs_database=1,
        ),
        example(
            "Show me an example SQL UPDATE statement, but do not run it.",
        ),
        example(
            "Read customer_ids.txt and query those customers in the database.",
            needs_filesystem=1,
            needs_database=1,
        ),
    ]


def communication_examples() -> list[dict]:
    return [
        example(
            "Find the latest email from the finance department.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Draft a message to the finance department but do not send it.",
        ),
        example(
            "Tell me what meetings I have tomorrow afternoon.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Explain how calendar invitations work.",
        ),
        example(
            "Find the latest invitation email and add the meeting to my calendar.",
            needs_network=1,
            needs_email=1,
            needs_calendar=1,
        ),
    ]


def composite_examples() -> list[dict]:
    return [
        example(
            "Run the test suite, inspect the failing source file, and identify the commit that introduced the failure.",
            needs_filesystem=1,
            needs_git=1,
            needs_shell=1,
        ),
        example(
            "Look up the current release notes online and save a summary locally.",
            needs_filesystem=1,
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Read the local customer IDs, query their records, and email the results to support.",
            needs_filesystem=1,
            needs_database=1,
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Inspect the last migration commit, run its tests, and check the resulting records in the development database.",
            needs_filesystem=1,
            needs_git=1,
            needs_shell=1,
            needs_database=1,
        ),
    ]


def scenario(
    templates: list[str],
    label_values: dict[str, int],
) -> list[dict]:
    items = []

    for text in templates:
        items.append(
            example(
                text,
                **label_values,
            )
        )

    return items


def hard_negative_examples() -> list[dict]:
    examples = []

    examples += scenario(
        [
            "Explain how Git branches work.",
            "Describe what a Git merge conflict is.",
            "Show me an example git reset command, but do not run it.",
            "Explain what git stash does.",
        ],
        {},
    )

    examples += scenario(
        [
            "Explain what an SQL JOIN does.",
            "Show an example DELETE query without executing it.",
            "Describe database transactions.",
            "Explain what database normalization means.",
        ],
        {},
    )

    examples += scenario(
        [
            "Draft an email asking for a project update, but do not send it.",
            "Write an example reply to a customer complaint.",
            "Explain how email threading works.",
            "Compose a short email confirming receipt of a document.",
        ],
        {},
    )

    examples += scenario(
        [
            "Explain how calendar reminders work.",
            "Show an example iCalendar event.",
            "Describe the difference between recurring and one-time events.",
            "Draft a meeting invitation without adding it to my calendar.",
        ],
        {},
    )

    examples += scenario(
        [
            "Explain what an HTTP request is.",
            "Show a curl example without executing it.",
            "Describe how DNS resolution works.",
            "Explain what a REST API is.",
        ],
        {},
    )

    return examples


def action_examples() -> list[dict]:
    examples = []

    examples += scenario(
        [
            "Read the local application log and summarize the last error.",
            "Open the local configuration file and report the retry limit.",
            "Save the generated report to a local Markdown file.",
            "Update the local settings file to enable verbose logging.",
        ],
        {
            "needs_filesystem": 1,
        },
    )

    examples += scenario(
        [
            "Run the unit test suite.",
            "Execute the local benchmark script.",
            "Run the build command and report whether it succeeds.",
            "Execute the migration tool in dry-run mode.",
        ],
        {
            "needs_shell": 1,
        },
    )

    examples += scenario(
        [
            "Inspect the Git history for the last change to auth.py.",
            "Create a Git branch named experiment/new-router.",
            "Compare the current branch with main.",
            "Find the commit that introduced the new configuration format.",
        ],
        {
            "needs_filesystem": 1,
            "needs_git": 1,
        },
    )

    examples += scenario(
        [
            "Search the official Python documentation for asyncio task groups.",
            "Open the current Rust documentation for async functions.",
            "Look up the latest stable Node.js documentation.",
            "Check the vendor documentation for the current API limits.",
        ],
        {
            "needs_browser": 1,
            "needs_network": 1,
        },
    )

    examples += scenario(
        [
            "Check whether the supplied API endpoint responds.",
            "Resolve the supplied hostname using the network.",
            "Download data from the supplied URL.",
            "Check whether the remote service is reachable.",
        ],
        {
            "needs_network": 1,
        },
    )

    examples += scenario(
        [
            "Query the customer database for inactive accounts.",
            "Find order 9917 in the database.",
            "Update the database status of job 551 to completed.",
            "Count failed payments in the database.",
        ],
        {
            "needs_database": 1,
        },
    )

    examples += scenario(
        [
            "Find the latest email from the customer.",
            "Reply to the most recent support email.",
            "Search my mailbox for the invoice message.",
            "Send an email confirming the deployment completed.",
        ],
        {
            "needs_network": 1,
            "needs_email": 1,
        },
    )

    examples += scenario(
        [
            "Find my next calendar event.",
            "Move tomorrow's team meeting to 14:00.",
            "Create a calendar event for Friday morning.",
            "Check whether I have a meeting at 16:00.",
        ],
        {
            "needs_network": 1,
            "needs_calendar": 1,
        },
    )

    return examples


def workflow_examples() -> list[dict]:
    return [
        example(
            "Read the local test configuration and run the test suite.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Inspect the Git history and run the tests from the previous commit.",
            needs_filesystem=1,
            needs_git=1,
            needs_shell=1,
        ),
        example(
            "Search the official documentation and save the result locally.",
            needs_filesystem=1,
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Download the supplied file and save it to the local project directory.",
            needs_filesystem=1,
            needs_network=1,
        ),
        example(
            "Read account IDs from a local file and query them in the database.",
            needs_filesystem=1,
            needs_database=1,
        ),
        example(
            "Query the database and email the resulting summary to the support team.",
            needs_database=1,
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Find the latest email from the vendor and schedule a follow-up meeting.",
            needs_network=1,
            needs_email=1,
            needs_calendar=1,
        ),
        example(
            "Read the release notes locally, search online for compatibility information, and email the summary.",
            needs_filesystem=1,
            needs_browser=1,
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Inspect the latest migration commit, run its tests, and query the development database for the migrated rows.",
            needs_filesystem=1,
            needs_git=1,
            needs_shell=1,
            needs_database=1,
        ),
        example(
            "Search the current documentation, update the local configuration file, and commit the change.",
            needs_filesystem=1,
            needs_git=1,
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Find the latest incident email, query the affected account, and add a follow-up meeting to the calendar.",
            needs_network=1,
            needs_database=1,
            needs_email=1,
            needs_calendar=1,
        ),
        example(
            "Inspect the repository, run the deployment script, and check the public health endpoint.",
            needs_filesystem=1,
            needs_git=1,
            needs_shell=1,
            needs_network=1,
        ),
    ]


def calendar_expansion_examples() -> list[dict]:
    return [
        example(
            "Check whether I have any meetings before noon tomorrow.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Create a calendar event for the deployment review next Tuesday.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Move my Friday afternoon appointment to Monday morning.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Delete the cancelled sprint review from my calendar.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Find my next free 30-minute slot this week.",
            needs_network=1,
            needs_calendar=1,
        ),
        example(
            "Show me an example calendar event in JSON without accessing my calendar.",
        ),
        example(
            "Explain what calendar availability means.",
        ),
        example(
            "Draft a meeting invitation but do not create the event.",
        ),
    ]


def email_expansion_examples() -> list[dict]:
    return [
        example(
            "Find the latest unread email from the project manager.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Reply to the latest invoice email and confirm receipt.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Forward the most recent support message to the engineering team.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Search my inbox for messages mentioning the deployment failure.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Send the release summary to the QA mailing list.",
            needs_network=1,
            needs_email=1,
        ),
        example(
            "Write an example email requesting access to the repository without sending it.",
        ),
        example(
            "Explain what email forwarding does.",
        ),
        example(
            "Draft a reply thanking the sender, but do not access my mailbox.",
        ),
    ]


def browser_expansion_examples() -> list[dict]:
    return [
        example(
            "Open the official Docker documentation and find the current Compose syntax.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Search the vendor website for the latest supported API version.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Check the current Python documentation for pathlib.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Look up the latest release notes on the project's website.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Search online for the current stable version of SQLite.",
            needs_browser=1,
            needs_network=1,
        ),
        example(
            "Explain what browser local storage is without opening a website.",
        ),
        example(
            "Describe what a browser tab is.",
        ),
        example(
            "Summarize this pasted web page text without opening a browser.",
        ),
    ]


def shell_expansion_examples() -> list[dict]:
    return [
        example(
            "Run the formatter and report which files would change.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Execute the project's integration tests.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Run the local build script and report its exit code.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Execute the benchmark command and report the elapsed time.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Run the dependency checker against the project.",
            needs_filesystem=1,
            needs_shell=1,
        ),
        example(
            "Show an example make command but do not execute it.",
        ),
        example(
            "Explain what chmod does.",
        ),
        example(
            "Describe how a shell pipeline works.",
        ),
    ]


def git_expansion_examples() -> list[dict]:
    return [
        example(
            "Inspect the repository and show the commits made since the last release tag.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Create a commit containing the current configuration changes.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Find which commit last modified router.py.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Check whether the current branch has uncommitted changes.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Compare the current working tree with the main branch.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Restore config.yaml from the previous commit.",
            needs_filesystem=1,
            needs_git=1,
        ),
        example(
            "Explain what git bisect does without inspecting a repository.",
        ),
        example(
            "Show an example git log command without running it.",
        ),
    ]


def database_expansion_examples() -> list[dict]:
    return [
        example(
            "Find all failed jobs created during the last hour in the database.",
            needs_database=1,
        ),
        example(
            "Update the user record for account 812 so it is marked active.",
            needs_database=1,
        ),
        example(
            "Delete expired session records from the development database.",
            needs_database=1,
        ),
        example(
            "Count orders grouped by their current status.",
            needs_database=1,
        ),
        example(
            "Look up the database record for customer 4812.",
            needs_database=1,
        ),
        example(
            "Check whether migration 42 has been applied in the database.",
            needs_database=1,
        ),
        example(
            "Explain what a database index is without querying anything.",
        ),
        example(
            "Show an example SQL transaction without executing it.",
        ),
    ]


def expand_with_paraphrases(
    examples: list[dict],
    variants_per_example: int = 3,
) -> list[dict]:
    expanded = []

    for item in examples:
        expanded.extend(
            paraphrase_variants(
                item,
                max_variants=variants_per_example,
            )
        )

    return deduplicate(expanded)


def build_candidates() -> list[dict]:
    examples = []

    examples.extend(filesystem_examples())
    examples.extend(shell_examples())
    examples.extend(git_examples())
    examples.extend(browser_network_examples())
    examples.extend(database_examples())
    examples.extend(communication_examples())
    examples.extend(composite_examples())

    examples.extend(hard_negative_examples())
    examples.extend(action_examples())
    examples.extend(workflow_examples())

    examples.extend(calendar_expansion_examples())
    examples.extend(email_expansion_examples())
    examples.extend(browser_expansion_examples())
    examples.extend(shell_expansion_examples())

    examples.extend(git_expansion_examples())
    examples.extend(database_expansion_examples())

    examples = deduplicate(examples)

    rng = random.Random(SEED)
    rng.shuffle(examples)

    return examples


def merge_training_examples(
    existing: list[dict],
    expanded: list[dict],
    blocked_states: set[str],
) -> list[dict]:
    combined = []

    for item in existing + expanded:
        state = item["state"].strip()

        if state in blocked_states:
            continue

        combined.append(item)

    return deduplicate(combined)


def convert_examples(examples):
    return [
        {
            "state": example.state,
            "labels": dict(example.labels),
        }
        for example in examples
    ]


def print_sample(
    examples: list[dict],
    count: int = 30,
    seed: int = SEED,
) -> None:
    rng = random.Random(seed)

    sample = rng.sample(
        examples,
        min(count, len(examples)),
    )

    print("\nSample expanded examples:\n")

    for index, item in enumerate(sample, start=1):
        active = [label for label, value in item["labels"].items() if value]

        print(f"{index:>2}. {item['state']}")
        print(
            "    labels:",
            ", ".join(active) if active else "none",
        )


def main() -> None:
    examples = build_candidates()

    # 1. Write curated candidates
    write_jsonl(
        OUTPUT_PATH,
        examples,
    )

    print(f"Wrote {len(examples)} curated candidates " f"to {OUTPUT_PATH}")

    print_coverage(examples)

    # 2. Expand curated candidates with safe paraphrases
    expanded = expand_with_paraphrases(
        examples,
        variants_per_example=3,
    )

    write_jsonl(
        EXPANDED_OUTPUT_PATH,
        expanded,
    )

    print(f"\nWrote {len(expanded)} expanded examples " f"to {EXPANDED_OUTPUT_PATH}")

    print_coverage(expanded)

    # 3. Print a random sample for manual inspection
    print_sample(
        expanded,
        count=30,
    )

    # 4. Load existing training/development/test datasets
    existing = convert_examples(load_routing_dataset("data/routing_v3.jsonl"))

    validation = load_routing_dataset("data/routing_v3_validation.jsonl")

    old_validation = load_routing_dataset("data/routing_benchmark.jsonl")

    final_test = load_routing_dataset("data/routing_final_test.jsonl")

    # 5. Build the protected-state set
    blocked_states = {
        example.state.strip()
        for dataset in (
            validation,
            old_validation,
            final_test,
        )
        for example in dataset
    }

    # 6. Merge existing training data with expanded V3 data
    training = merge_training_examples(
        existing=existing,
        expanded=expanded,
        blocked_states=blocked_states,
    )

    # 7. Write final V3 training corpus
    write_jsonl(
        TRAINING_OUTPUT_PATH,
        training,
    )

    print(
        f"\nWrote {len(training)} final V3 training examples "
        f"to {TRAINING_OUTPUT_PATH}"
    )

    print_coverage(training)

    # 8. Final overlap checks
    training_states = {item["state"].strip() for item in training}

    v3_validation_states = {example.state.strip() for example in validation}

    old_validation_states = {example.state.strip() for example in old_validation}

    final_test_states = {example.state.strip() for example in final_test}

    print("\nOverlap checks:")

    print(
        "V3 validation overlap:",
        len(training_states & v3_validation_states),
    )

    print(
        "Old benchmark overlap:",
        len(training_states & old_validation_states),
    )

    print(
        "Final test overlap:",
        len(training_states & final_test_states),
    )

    assert not (training_states & v3_validation_states)

    assert not (training_states & old_validation_states)

    assert not (training_states & final_test_states)

    print("\nAll overlap checks passed.")


if __name__ == "__main__":
    main()
