import logging
import time


logger = logging.getLogger()


def run_until_no_prompt_errors(
    prompts,
    run_once_fn,
    label,
    max_rounds=0,
    sleep_seconds=0.0,
):
    """Retry failed LLM profile prompts until every prompt succeeds.

    Args:
        prompts: dict mapping id -> prompt.
        run_once_fn: callable(pending_prompts) -> (ok_dict, err_dict).
        label: text used in logs.
        max_rounds: 0 means unlimited rounds.
        sleep_seconds: optional delay between repair rounds.
    """
    if not prompts:
        return {}, {}

    pending = dict(prompts)
    success = {}
    round_idx = 0

    while pending:
        round_idx += 1
        logger.info(
            "[DATASET][REPAIR] %s round %d: processing %d pending prompts.",
            label,
            round_idx,
            len(pending),
        )
        ok, err = run_once_fn(pending)
        success.update(ok)

        missing = {key: pending[key] for key in pending.keys() if key not in ok and key not in err}
        pending = {key: prompts[key] for key in err.keys()}
        pending.update(missing)

        if not pending:
            logger.info(
                "[DATASET][REPAIR] %s completed after %d round(s); all prompts succeeded.",
                label,
                round_idx,
            )
            return success, {}

        logger.warning(
            "[DATASET][REPAIR] %s still has %d failed prompt(s) after round %d; retrying only failures.",
            label,
            len(pending),
            round_idx,
        )
        if max_rounds and round_idx >= max_rounds:
            raise RuntimeError(
                f"{label} still has {len(pending)} failed prompt(s) after {round_idx} repair rounds."
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return success, {}
