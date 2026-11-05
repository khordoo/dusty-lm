"""Subscription-only Codex transport and credential-free offline tool verification."""

import argparse
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .controller import Controller, initialize, write_json
from .datasets import DEVELOPMENT
from .source import ARTIFACTS, sha256

PACKAGE = Path(__file__).parent
PIN = PACKAGE / "subscription_pin.json"
TOOL_METADATA = {
    "tool_mode": "direct",
    "apply_patch_tool_type": None,
    "experimental_supported_tools": [],
    "supports_search_tool": False,
}
OFF_FEATURES = """code_mode code_mode_only code_mode_host collaboration_modes shell_tool unified_exec unified_exec_tty shell_snapshot view_image image_generation multi_agent multi_agent_v2 apps plugins remote_plugin browser_use browser_use_external browser_use_full_cdp_access in_app_browser computer_use skill_search skill_mcp_dependency_install sleep_tool tool_suggest goals hooks worktrees workspace_dependencies memories current_time_reminder default_mode_request_user_input send_async_message send_message_to_user_async""".split()
FORBIDDEN_ENV = {
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "CODEX_MODEL_PROVIDER",
    "CHATGPT_BASE_URL",
}


def cli_identity():
    executable = shutil.which("codex")
    if not executable:
        raise ValueError("Codex CLI required")
    version = subprocess.check_output([executable, "--version"], text=True).strip()
    identity = {"version": version, "binary_sha256": sha256(Path(executable).resolve())}
    if not PIN.exists() or json.loads(PIN.read_text())["cli"] != identity:
        raise ValueError(
            "CLI version/binary differs from reviewed pin; reverify configuration first"
        )
    return identity


def bundled_model():
    result = subprocess.check_output(["codex", "debug", "models", "--bundled"], text=True)
    models = json.loads(result)["models"]
    model = next(m for m in models if m["slug"] == "gpt-6-astra")
    expected = json.loads(PIN.read_text())["bundled_astra_sha256"]
    import hashlib

    if hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest() != expected:
        raise ValueError("Bundled Astra catalog changed")
    modified = dict(model, **TOOL_METADATA)
    if {k for k in model if model[k] != modified[k]} != set(TOOL_METADATA):
        raise ValueError("Unexpected metadata diff")
    return model, modified


def subscription_auth():
    rejected = sorted(k for k in FORBIDDEN_ENV if os.environ.get(k))
    if rejected:
        raise ValueError(
            "Subscription-only mode rejects API keys/provider overrides: " + ", ".join(rejected)
        )
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    path = home / "auth.json"
    data = json.loads(path.read_text())
    if data.get("auth_mode") != "chatgpt" or data.get("OPENAI_API_KEY") or not data.get("tokens"):
        raise ValueError("Existing ChatGPT subscription sign-in required; no API-key fallback")
    return path.resolve()


def configuration(home, catalog):
    return {
        "model": "gpt-6-astra",
        "model_provider": "openai",
        "model_reasoning_effort": "high",
        "forced_login_method": "chatgpt",
        "approval_policy": "never",
        "sandbox_mode": "read-only",
        "web_search": "disabled",
        "model_instructions_file": str(home / "instructions.md"),
        "model_catalog_json": str(catalog),
        "project_doc_max_bytes": 0,
        "include_apps_instructions": False,
        "include_collaboration_mode_instructions": False,
        "include_permissions_instructions": False,
        "agents.enabled": False,
        "tools.update_plan.enabled": False,
        "tools.experimental_request_user_input.enabled": False,
        "skills.include_instructions": False,
        "skills.bundled.enabled": False,
        "features.skip_host_skill_discovery": True,
        "features.omit_skill_descriptions": True,
        **{"features." + name: False for name in OFF_FEATURES},
    }


def config_args(values):
    args = []
    for key, value in values.items():
        args.extend(["-c", key + "=" + json.dumps(value)])
    return args


def clean_environment(home):
    # No ambient API keys, provider settings, plugin endpoints or inherited tool hosts.
    result = {
        k: os.environ[k]
        for k in ("PATH", "LANG", "TZ", "TMPDIR", "SSL_CERT_FILE")
        if k in os.environ
    }
    result.update(HOME=str(home), CODEX_HOME=str(home))
    return result


def prepare_home(directory, auth=None):
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    original, modified = bundled_model()
    catalog = directory / "models.json"
    write_json(catalog, {"models": [modified]})
    (directory / "instructions.md").write_text("Answer the user's request.\n")
    if auth:
        # Credential material is not copied, printed, included in prompts or Docker mounts.
        (directory / "auth.json").symlink_to(auth)
    values = configuration(directory, catalog)
    write_json(directory / "effective_overrides.json", values)
    return values


def visible_tools(input_items):
    found = []
    for item in input_items:
        if item.get("type") == "additional_tools":
            found.extend(item.get("tools", []))
    return found


def offline_probe():
    identity = cli_identity()
    with tempfile.TemporaryDirectory(prefix="milo-codex-offline-") as temporary:
        root = Path(temporary)
        home = root / "home"
        config = prepare_home(home)
        work = root / "public"
        work.mkdir()
        environment = clean_environment(home)
        args = config_args(config)
        # Debug builds exactly the model-visible input without sampling a model.
        debug = subprocess.run(
            ["codex", "debug", "prompt-input", *args, "Offline prompt probe."],
            cwd=work,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        prompt = json.loads(debug.stdout)
        if visible_tools(prompt) or "<skills_instructions>" in debug.stdout:
            raise ValueError("Offline prompt exposes tools or skill instructions")
        captured = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                captured.append(json.loads(raw))  # Never record HTTP headers/credentials.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {
                        "type": "response.created",
                        "response": {
                            "id": "offline",
                            "object": "response",
                            "status": "in_progress",
                            "output": [],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": '{"offline_probe":true}'}],
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "offline",
                            "object": "response",
                            "status": "completed",
                            "output": [],
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        },
                    },
                ]
                for event in events:
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        # This test alone overrides the provider to a credential-free localhost
        # mock. No request is forwarded and no remote model can be sampled.
        mock = dict(config)
        mock.pop("forced_login_method")
        mock.update(
            {
                "model_provider": "milo_offline_mock",
                "model_providers.milo_offline_mock.name": "offline mock",
                "model_providers.milo_offline_mock.base_url": f"http://127.0.0.1:{server.server_address[1]}",
                "model_providers.milo_offline_mock.wire_api": "responses",
                "model_providers.milo_offline_mock.requires_openai_auth": False,
                "model_providers.milo_offline_mock.request_max_retries": 0,
                "model_providers.milo_offline_mock.stream_max_retries": 0,
            }
        )
        try:
            completed = subprocess.run(
                [
                    "codex",
                    "--no-daemon",
                    "exec",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--strict-config",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--json",
                    *config_args(mock),
                    "Offline wire probe.",
                ],
                cwd=work,
                env=environment,
                input="",
                capture_output=True,
                text=True,
                timeout=30,
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
        if completed.returncode or len(captured) != 1:
            raise ValueError("Offline mock failed: " + completed.stderr[-1000:])
        request = captured[0]
        if request.get("tools") or visible_tools(request.get("input", [])):
            raise ValueError("Wire request exposes tools")
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        inspect_events(events)
        report = {
            "cli": identity,
            "offline_prompt_tool_count": 0,
            "offline_wire_tool_count": 0,
            "skill_instructions_present": False,
            "model": "gpt-6-astra",
            "reasoning": "high",
            "metadata_changes": TOOL_METADATA,
            "all_other_metadata_preserved": True,
            "external_model_calls": 0,
            "live_subscription_call_verified": False,
            "limitation": "Localhost mock uses credential-free provider; live ChatGPT backend behavior not yet tested.",
            "timestamp": time.time(),
        }
    write_json(ARTIFACTS / "subscription_preflight.json", report)
    return report


def inspect_events(events):
    allowed = {"agent_message", "reasoning", "error"}
    if any(e.get("type") in {"error", "turn.failed"} for e in events):
        raise ValueError("CLI turn failed; no request execution/retry")
    for event in events:
        item = event.get("item", {})
        if item.get("type") == "error" and not item.get("message", "").startswith(
            "Under-development features enabled:"
        ):
            raise ValueError("Unexpected CLI error item")
        if item and item.get("type") not in allowed:
            raise ValueError("Unexpected CLI tool/event: " + str(item.get("type")))
    if not any(e.get("type") == "turn.completed" for e in events):
        raise ValueError("No completed CLI turn")
    return [e["usage"] for e in events if e.get("type") == "turn.completed" and "usage" in e]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", action="store_true", help="Credential-free no-model offline verification"
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Explicit subscription model-run gate"
    )
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(offline_probe(), indent=2))
        return
    if not args.execute or args.run is None:
        raise SystemExit(
            "Subscription editor not started. Use --probe for no-model verification; --execute and a fresh --run are required later."
        )
    auth = subscription_auth()
    cli_identity()
    from .launch import validate_launch_setup

    validate_launch_setup()
    offline_probe()
    state = initialize(args.run)
    run = args.run.resolve()
    home = run / "codex-private"
    config = prepare_home(home, auth)
    public = run / "editor-public"
    public.mkdir(mode=0o700)
    # Only known public data enters the first message. No repository/private-suite access.
    import torch
    from tokenizers import Tokenizer

    tensors = torch.load(Path(state["source"]) / "model.pt", weights_only=True, map_location="cpu")
    if "model_state_dict" in tensors:
        tensors = tensors["model_state_dict"]
    tokenizer = Tokenizer.from_file(str(Path(state["source"]) / "tokenizer.json"))
    metadata = {
        "tensors": {
            name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in tensors.items()
        },
        "name_tokenization": {
            s: {"ids": tokenizer.encode(s).ids, "tokens": tokenizer.encode(s).tokens}
            for s in [" dusty", " milo", "dusty", "milo"]
        },
    }
    message = (
        PACKAGE.joinpath("editor_prompt.md").read_text()
        + "\n"
        + json.dumps(
            {
                "protocol": state["protocol"],
                "development": json.loads(DEVELOPMENT.read_text()),
                "source_metadata": metadata,
            }
        )
    )
    write_json(
        run / "subscription_settings.json",
        {
            "cli": cli_identity(),
            "transport": "ChatGPT subscription via Codex",
            "model": "gpt-6-astra",
            "reasoning": "high",
            "tools": [],
            "api_fallback": False,
            "prompt_sha256": sha256(PACKAGE / "editor_prompt.md"),
            "max_turns": 180,
            "usage_accounting": "Raw CLI turn usage recorded; do not sum cumulative counters blindly.",
        },
    )
    controller = Controller(run)
    thread = None
    for turn in range(180):
        # Same pinned catalog/config on every resume; never auto-reset the conversation.
        answer = home / f"answer-{turn:03d}.txt"
        command = [
            "codex",
            "--no-daemon",
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--skip-git-repo-check",
            "--json",
            *config_args(config),
            "-o",
            str(answer),
        ]
        command += ["resume", thread, "-"] if thread else ["-"]
        completed = subprocess.run(
            command,
            cwd=public,
            env=clean_environment(home),
            input=message,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        write_json(home / f"events-{turn:03d}.json", events)
        (home / f"stderr-{turn:03d}.log").write_text(completed.stderr)
        if completed.returncode:
            raise SystemExit("CLI failed; no retry, model substitution or API fallback")
        inspect_events(events)
        started = [e["thread_id"] for e in events if e.get("type") == "thread.started"]
        if thread is None:
            if len(started) != 1:
                raise SystemExit("No unique fresh thread ID")
            thread = started[0]
        elif started and any(t != thread for t in started):
            raise SystemExit("Conversation changed unexpectedly")
        try:
            result = controller.dispatch(json.loads(answer.read_text()))
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result = {"error": str(error)}
        if result.get("final"):
            print(
                json.dumps(
                    {
                        "final": result["final"],
                        "transport": "ChatGPT subscription",
                        "turns": turn + 1,
                    }
                )
            )
            return
        message = json.dumps(result)
    raise SystemExit("Turn budget exhausted; experiment incomplete")


if __name__ == "__main__":
    main()
