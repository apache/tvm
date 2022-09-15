import json
import statistics
from typing import List, Dict, Any
from git_utils import get
import scipy.stats
import re
import logging


MAIN_INFO_URL = "https://ci.tlcpack.ai/blue/rest/organizations/jenkins/pipelines/tvm/runs/?branch=main&start=0&limit=26"


def find_target_url(pr_head: Dict[str, Any]) -> str:
    for status in pr_head["statusCheckRollup"]["contexts"]["nodes"]:
        if status.get("context", "") == "tvm-ci/pr-head":
            return status["targetUrl"]

    raise RuntimeError(f"Unable to find tvm-ci/pr-head status in {pr_head}")


def fetch_past_build_times_s() -> List[float]:
    data = get(MAIN_INFO_URL).decode()
    data = json.loads(data)
    build_times_s = []
    logging.info(f"Fetched {len(data)} builds from main")
    for item in data:
        # Only look at completed builds
        if not can_use_build(item):
            logging.info("Skipping failed build")
            continue

        duration = item["durationInMillis"]
        build_times_s.append(duration / 1000.0)

    return build_times_s


def can_use_build(build: Dict[str, Any]):
    return build["state"] == "FINISHED" and build["result"] == "SUCCESS"


def fetch_build_time_s(branch: str, build: str) -> float:
    build = int(build)
    info_url = f"https://ci.tlcpack.ai/blue/rest/organizations/jenkins/pipelines/tvm/runs/?branch={branch}&start=0&limit=25"
    data = get(info_url).decode()
    data = json.loads(data)

    for item in data:
        if item["id"] == str(build):
            if can_use_build(item):
                return item["durationInMillis"] / 1000.0
            else:
                raise RuntimeError(
                    f"Found build for {branch} with {build} but cannot use it: {item}"
                )

    raise RuntimeError(f"Unable to find branch {branch} with {build} in {data}")


def ci_runtime_comment(pr: Dict[str, Any]) -> str:
    pr_head = pr["commits"]["nodes"][0]["commit"]
    target_url = find_target_url(pr_head)
    logging.info(f"Got target url {target_url}")
    m = re.search(r"/job/(PR-\d+)/(\d+)", target_url)
    branch, build = m.groups()

    logging.info(f"Calculating CI runtime for {branch} with {build}")
    main_build_times_s = fetch_past_build_times_s()
    if len(main_build_times_s) == 0:
        logging.info("Found no usable builds on main, quitting")
        return None
    x = statistics.mean(main_build_times_s)
    logging.info(f"Sample mean from main: {x}")
    current_build_time_s = fetch_build_time_s(branch=branch, build=build)
    build_url = (
        f"https://ci.tlcpack.ai/blue/organizations/jenkins/tvm/detail/{branch}/{build}/pipeline"
    )
    res = scipy.stats.ttest_1samp(main_build_times_s, current_build_time_s)
    logging.info(f"t-stats: {res}")
    change = -(x - current_build_time_s) / x * 100.0
    change = round(change, 2)
    if res.pvalue < 0.05:
        return f"This PR **significantly changed [CI runtime]({build_url}): {change}%**"
    else:
        return f"This PR had no significant effect on [CI runtime]({build_url}): {change}%"
