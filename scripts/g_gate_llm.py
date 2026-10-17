# ruff: noqa: RUF001
"""Local-LLM style-material gate (offline research; synthetic data only).

Modes: ``direct`` (one 是/否 judgement, p_yes from first-token logits),
``content_only`` (premise conflict only), ``ga`` (D18 two-stage: acceptable
foci/actions from the current dialogue, then per-candidate action + content) and
``ga2`` (D20: an offline candidate card cached per c/r text, its own inclusive
stage-1 foci, then a per-scene premise/permission/sufficiency check combined by ``combine2``).
The backend is injected; only ``MlxBackend`` imports mlx, and only ``main``
constructs it. Run real inference with ``~/.rcpem/venv/bin/python``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

MODES = ("direct", "content_only", "ga", "ga2")
CONTENT_VALUES = ("compatible", "conflict", "unknown")
PERMISSION_VALUES = ("ok", "violation", "unknown")
MAX_TOKENS = {"stage1": 256, "stage2": 192, "content_only": 4, "card": 384, "judge2": 128}
CONTENT_WORDS = {"兼容": "compatible", "冲突": "conflict", "未知": "unknown"}

LABELS = {
    "five": {
        "A": "接话回应：直接回应她这句话，回答、附和、评价、夸、安慰；针对她刚说的话反问也算",
        "B": "追问关心：向她提新问题，或关心叮嘱",
        "C": "说自己的事：报备或分享自己的事",
        "D": "调侃：开玩笑、损她、互怼、搞怪；边接话边损算调侃",
        "E": "撒娇：示弱、假装委屈、要关注、表白",
    },
    "four": {
        "A": "接话/追问：直接回应她这句话（回答、附和、评价、夸、安慰，针对她刚说的话反问也算），"
             "或向她提新问题、关心叮嘱",
        "C": "说自己的事：报备或分享自己的事",
        "D": "调侃：开玩笑、损她、互怼、搞怪；边接话边损算调侃",
        "E": "撒娇：示弱、假装委屈、要关注、表白",
    },
}

ROLE_NOTE = "对话里“她”是女朋友，“他”是主人。机器人模仿主人回复她。"

DIRECT_SYSTEM = (
    "你在为陪伴机器人筛选历史对话参考示例。" + ROLE_NOTE + "\n"
    "给你【当前对话】和一条【参考示例】（当时她说的话，以及主人当时的全部回复）。"
    "参考示例只用来让机器人学习回应思路和说话方式，不会被原样照说；"
    "但示例回复里的回应动作、事实、承诺和指代会被机器人模仿。\n"
    "判断：把这条参考示例给机器人参考来回复当前对话，是否合适？\n"
    "合适：示例的回应方式能自然迁移到当前对话（话题可以不同），"
    "并且示例回复的每一条消息都不和当前对话的事实、状态、需求或许可冲突。\n"
    "不合适：当前事情已经完成或被更正而示例还按旧状态回应；当前需求与示例相反；"
    "示例针对的人物或主体与当前不符；示例回复带有当前不存在的事实或承诺；"
    "许可不允许示例中的称呼或玩笑；示例的回应方式在当前对话里不成立。\n"
    "只回答一个字：是 或 否。"
)

CONTENT_SYSTEM = (
    "你在为陪伴机器人检查历史对话参考示例的内容前提。" + ROLE_NOTE + "\n"
    "给你【当前对话】和一条【参考示例】（当时她说的话，以及主人当时的全部回复）。"
    "不评价回应方式好不好，只检查示例回复的每一条消息的内容前提是否和当前对话冲突。\n"
    "冲突的情况：当前事情已经完成或被更正，示例还按旧状态说；当前需求与示例相反；"
    "示例针对的人物或主体与当前不符；示例回复带有当前对话不存在的旧事实或旧承诺；"
    "许可不允许示例中的称呼或玩笑。\n"
    "没有以上冲突就是兼容，话题不同本身不算冲突。信息不足以判断时答未知。\n"
    "只回答一个词：兼容、冲突 或 未知。"
)


def _label_block(labels: str) -> str:
    return "\n".join(f"{code}：{text}" for code, text in LABELS[labels].items())


def stage1_system(labels: str) -> str:
    codes = "、".join(LABELS[labels])
    return (
        "你在分析陪伴机器人当前该怎么回应。" + ROLE_NOTE + "\n"
        "只看【当前对话】和【许可】，列出机器人这轮回复可以承接的焦点（她这轮话里可回应的事，"
        "可以是整体），每个焦点列出此刻可以接受的回应动作类别（可多选，所有说得通的都列上）。\n"
        "动作类别：\n" + _label_block(labels) + "\n"
        "要求：焦点只引用当前对话里出现的信息，不补充没说过的事实；通常只列 1 个整体焦点，"
        "她说了几件不同的事时才分开列，最多 3 个，每个不超过 20 字；"
        "尊重她的明确要求和许可限制，不合适的动作不要列；"
        "无法判断时 actions 写 [\"unknown\"]。\n"
        "只输出一行紧凑的 JSON 对象，不要其他文字，格式："
        '{"foci":[{"id":"f1","focus":"……","actions":["' + codes.split("、")[0] + '"]}]}'
    )


def stage2_system(labels: str, focus_ids: list[str], with_permission: bool) -> str:
    permission_rule = (
        "permission：示例回复在当前【许可】下是否允许，ok / violation / unknown。\n"
        if with_permission else ""
    )
    permission_key = '"permission":"ok|violation|unknown",' if with_permission else ""
    codes = "|".join([*LABELS[labels], "unknown"])
    content_keys = ",".join(f'"{fid}":"compatible|conflict|unknown"' for fid in focus_ids)
    return (
        "你在检查一条历史对话参考示例能否给陪伴机器人参考。" + ROLE_NOTE + "\n"
        "参考示例只用来学习回应思路和说话方式，不会被原样照说；"
        "但示例回复里的每条消息都默认了一些前提（事情处于什么状态、她想要什么、说的是谁、"
        "做过什么承诺），机器人借鉴时会把这些前提带到当前对话。\n"
        "动作类别：\n" + _label_block(labels) + "\n"
        "请依次给出：\n"
        "t：只看【参考示例】本身，写出主人当时那几条回复默认的前提（逐条消息），"
        "不要描述当前对话，不超过 30 字。\n"
        "a：示例中主人回复的首要回应动作类别（结合当时她说的话判断），一个类别代码或 unknown。\n"
        + permission_rule +
        "content：对每个焦点逐一判断：机器人借鉴这条示例来回应该焦点时，"
        "把 t 里的每条前提放到【整个当前对话】里核对（不只核对这个焦点）："
        "任一前提与当前对话的任何内容相矛盾就是 conflict，比如示例默认事情还没做而当前已经做完、"
        "示例默认她想要某样东西而她说了不要、示例说的是她而当前说的是别人、"
        "示例带着当前对话里没有的旧事实或旧承诺；"
        "前提都成立、只是话题不同就是 compatible；当前对话信息不足以核对时是 unknown。"
        "示例与当前对话任何一处矛盾，所有焦点都是 conflict。\n"
        "只输出一行紧凑的 JSON 对象，不要其他文字，content 必须包含每个焦点，格式：\n"
        '{"t":"前提：……","a":"' + codes + '",' + permission_key
        + '"content":{' + content_keys + "}}"
    )


def stage1_ga2_system(labels: str) -> str:
    codes = "、".join(LABELS[labels])
    return (
        "你在分析陪伴机器人当前可以怎么回应。" + ROLE_NOTE + "\n"
        "只看【当前对话】和【许可】，列出机器人这轮回复可以承接的焦点（她这轮话里可回应的事，"
        "可以是整体），每个焦点列出此刻可以接受的回应动作类别。\n"
        "动作类别：\n" + _label_block(labels) + "\n"
        "要求：逐一考虑全部类别（" + codes + "），凡是在当前对话和许可下说得通的都列上："
        "先默认全部类别都列上，再只删去三种：她明确拒绝的、许可不允许的、"
        "和她当前状态明显不合的（例如她在难过时调侃她）；其余一律保留。"
        "焦点只引用当前对话里出现的信息，不补充没说过的事实；通常只列 1 个整体焦点，"
        "她说了几件不同的事时才分开列，最多 3 个，每个不超过 20 字；"
        "无法判断时 actions 写 [\"unknown\"]。\n"
        "只输出一行紧凑的 JSON 对象，不要其他文字，格式："
        '{"foci":[{"id":"f1","focus":"……","actions":["……"]}]}'
    )


PREMISE_KINDS = ("state", "need", "person", "old_fact", "promise")
PREMISE_VALUES = ("supported", "contradicted", "not_given")
CARD_PROMPT_VERSION = "ga2-card-v13"


def card_system(labels: str) -> str:
    codes = "|".join(LABELS[labels])
    advise = next(code for code, text in LABELS[labels].items() if "关心叮嘱" in text)
    return (
        "你在为陪伴机器人整理一条历史对话参考示例的卡片。" + ROLE_NOTE + "\n"
        "给你【参考示例】：当时她说的话，以及主人当时的全部回复（按顺序编号）。"
        "只看这条示例本身。卡片之后会拿去和别的对话比较，话题往往完全不同。\n"
        "动作类别：\n" + _label_block(labels) + "\n"
        "请输出：\n"
        "msgs：只针对主人的回复，共几条就写几项，顺序和编号一致，她说的话不算；"
        "evidence 先抄这条回复原文（不超过 20 字），"
        "action 再只根据这条回复本身按顺序判断：向她提问、关心她、给她出主意、"
        "提议下一步做什么（常以“吧”“吗”“？”结尾）→" + advise + "；"
        "只讲主人自己的经历或安排、和她无关→C；明显在损她、笑话她→D，语气轻松不算；"
        "示弱、要关注、表白→E；其余直接回应她的话→A。\n"
        "persons：回复涉及的人物，用“她”“他”“别人”这样的槽位，不写名字。\n"
        "needs_permission：nickname 回复里是否给她起昵称，tease 回复里是否调侃、损她。\n"
        "premises：别的对话借鉴主人这几条回复时必须成立的前提，1 到 5 条。"
        "按回复顺序写：先写第1条回复默认的前提，再写第2条的，依次类推，每条回复至少一条，"
        "几条回复默认同一件事时可以合写一条。"
        "写每条消息默认的情况，不猜回复以外的事；问句和建议写它默认已经成立的情况，"
        "不写问句本身（例如“这周末一起去吧”默认“她想去做这件事”）。"
        "重点写回复依赖的关键状态：结果是好是坏、事情做没做完、她想不想要、说的是谁、"
        "她以后还想不想做。"
        "每条不超过 25 字，必须写成换个话题也能核对的抽象说法：不出现示例里的具体物品、"
        "活动、地点、数量、钱数，一律换成“这件事”“那样东西”，"
        "但要保留关键区别（是她的还是别人的、旧的还是新的、做了还是没做、好还是坏），"
        "例如回复“恭喜你考过了”的前提写“这件事的结果是好的”。"
        "再强调：前提只写抽象状态（例如“她做完了这件事”“她不想要那样东西”），"
        "不写具体名词。"
        "kind：state 事情处于什么状态，need 她想要什么，person 说的是谁、谁做的，"
        "old_fact 回复里提到的、换个话题就不一定有的具体事实，promise 主人做过的承诺。\n"
        "只输出一行紧凑的 JSON 对象，不要其他文字，msgs、persons、needs_permission、premises "
        "四个键都必须有，格式：\n"
        '{"msgs":[{"i":1,"evidence":"……","action":"' + codes + '"}],"persons":["她"],'
        '"needs_permission":{"nickname":false,"tease":false},'
        '"premises":[{"id":"p1","text":"……","kind":"state|need|person|old_fact|promise"}]}'
    )


def judge2_system(premise_ids: list[str], with_permission: bool) -> str:
    permission_rule = (
        "permission：示例回复如果有昵称或调侃，看【许可】是否允许："
        "允许或回复里没有昵称和调侃为 ok，许可明确不允许为 violation，许可没说清为 unknown。\n"
        if with_permission else ""
    )
    permission_key = '"permission":"ok|violation|unknown",' if with_permission else ""
    premise_keys = ",".join(f'"{pid}":"supported|contradicted|not_given"' for pid in premise_ids)
    return (
        "你在核对一条历史对话参考示例能否给陪伴机器人参考。" + ROLE_NOTE + "\n"
        "机器人借鉴示例时，会把【示例前提】带到【当前对话】里。话题不同没关系，"
        "前提里的“这件事”“那样东西”换成当前对话里的事来理解。\n"
        + permission_rule +
        "premises：每条前提放到整个当前对话里核对，只能填 supported、contradicted、not_given：\n"
        "先找出当前对话里和这条前提说的是同一方面的那句话，再判断：\n"
        "supported：当前对话里有直接对应的内容，同一主体、同一状态或需求，不能靠猜；"
        "当前对话只说了相关的事、没明确说出这个状态，是 not_given。"
        "只是同类的事不算：例如前提“她的考试通过了”而当前说“她弟弟的考试通过了”，"
        "前提“她把那样东西修好了”而当前说“她把那样东西换成新的了”，"
        "这类按情况判 contradicted 或 not_given，不能判 supported；\n"
        "contradicted：那句话说的情况和前提不一样，比如前提说结果好而当前说结果不好、"
        "前提说还没做而当前说已经做完、前提说她想要而当前说她不要、"
        "前提说是她的事而当前是别人的事或别的东西、前提里的事实或承诺被她否认；\n"
        "not_given：当前对话没有说，既不能确认也不能否认。"
        "当前对话没提到的旧事实、旧承诺是 not_given。\n"
        "key_missing：只看她这轮说的事情：关键的结果或状态她还没说出来"
        "（比如只说结果出来了、没说过没过；只说去看了、没说怎么样），"
        "补成好和坏两种情况会让这条示例一个适用一个不适用，为 true；"
        "她已经把结果或状态说清楚了，为 false。\n"
        "只输出一行紧凑的 JSON 对象，不要其他文字，premises 必须包含每条前提，"
        + ("permission、premises、key_missing 三个键" if with_permission
           else "premises、key_missing 两个键") + "都必须有，格式：\n"
        "{" + permission_key + '"premises":{' + premise_keys + '},"key_missing":false}'
    )


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(line, str) for line in value)


def load_events(path: Path) -> list[dict]:
    events, seen = [], set()
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
                raise ValueError("invalid event")
            if event["event_id"] in seen:
                raise ValueError("duplicate event id")
            seen.add(event["event_id"])
            if not _string_list(event.get("context")) or not isinstance(
                event.get("candidates"), list,
            ):
                raise ValueError("invalid event text")
            permission = event.get("permission")
            if permission is not None and not isinstance(permission, dict):
                raise ValueError("invalid permission")
            ids = set()
            for candidate in event["candidates"]:
                if not isinstance(candidate, dict) or not isinstance(candidate.get("cand_id"), str):
                    raise ValueError("invalid candidate")
                if candidate["cand_id"] in ids:
                    raise ValueError("duplicate candidate id")
                ids.add(candidate["cand_id"])
                if not _string_list(candidate.get("c")) or not _string_list(candidate.get("r")):
                    raise ValueError("invalid candidate text")
            events.append(event)
    return events


def permission_text(permission: dict | None) -> str:
    if permission is None:
        return "无额外许可限制。"
    return json.dumps(permission, ensure_ascii=False)


def render_context(event: dict) -> str:
    return "【当前对话】\n" + "\n".join(event["context"]) + "\n【许可】\n" + permission_text(
        event.get("permission"),
    )


def render_candidate(candidate: dict) -> str:
    replies = "\n".join(f"{i}. {line}" for i, line in enumerate(candidate["r"], 1))
    return "【参考示例】\n当时：\n" + "\n".join(candidate["c"]) + "\n主人当时回复：\n" + replies


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def direct_messages(event: dict, candidate: dict) -> list[dict]:
    return _messages(DIRECT_SYSTEM, render_context(event) + "\n" + render_candidate(candidate))


def content_messages(event: dict, candidate: dict) -> list[dict]:
    return _messages(CONTENT_SYSTEM, render_context(event) + "\n" + render_candidate(candidate))


def stage1_messages(event: dict, labels: str) -> list[dict]:
    return _messages(stage1_system(labels), render_context(event))


def stage1_ga2_messages(event: dict, labels: str) -> list[dict]:
    return _messages(stage1_ga2_system(labels), render_context(event))


def stage2_messages(event: dict, candidate: dict, foci: list[dict], labels: str) -> list[dict]:
    focus_lines = "\n".join(f"{f['id']}：{f['focus']}" for f in foci)
    user = render_candidate(candidate) + "\n" + render_context(event) + "\n【焦点】\n" + focus_lines
    system = stage2_system(labels, [f["id"] for f in foci], event.get("permission") is not None)
    return _messages(system, user)


def render_card_source(candidate: dict) -> str:
    replies = "\n".join(f"第{i}条：{line}" for i, line in enumerate(candidate["r"], 1))
    return ("【参考示例】\n她当时说：\n" + "\n".join(candidate["c"])
            + f"\n主人当时的回复（共 {len(candidate['r'])} 条）：\n" + replies)


def card_messages(candidate: dict, labels: str) -> list[dict]:
    return _messages(card_system(labels), render_card_source(candidate))


def render_premises(card: dict) -> str:
    return "【示例前提】\n" + "\n".join(f"{p['id']}：{p['text']}" for p in card["premises"])


def judge2_messages(event: dict, candidate: dict, card: dict, foci: list[dict]) -> list[dict]:
    focus_lines = "\n".join(f"{f['id']}：{f['focus']}" for f in foci)
    user = (render_candidate(candidate) + "\n" + render_premises(card) + "\n"
            + render_context(event) + "\n【焦点】\n" + focus_lines)
    system = judge2_system([p["id"] for p in card["premises"]], event.get("permission") is not None)
    return _messages(system, user)


def _json_object(text: str) -> dict:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return value


def parse_stage1(text: str, labels: str) -> list[dict]:
    value = _json_object(text)
    if set(value) != {"foci"} or not isinstance(value["foci"], list) or not value["foci"]:
        raise ValueError("invalid stage1 keys")
    codes, foci, ids = set(LABELS[labels]), [], set()
    for item in value["foci"]:
        if not isinstance(item, dict) or set(item) != {"id", "focus", "actions"}:
            raise ValueError("invalid focus")
        fid, focus, actions = item["id"], item["focus"], item["actions"]
        if not isinstance(fid, str) or not fid or fid in ids or not isinstance(focus, str):
            raise ValueError("invalid focus id")
        if not _string_list(actions) or not actions or len(set(actions)) != len(actions):
            raise ValueError("invalid actions")
        if actions != ["unknown"] and not set(actions) <= codes:
            raise ValueError("invalid action code")
        ids.add(fid)
        foci.append({"id": fid, "focus": focus, "actions": actions})
    return foci


def parse_stage2(text: str, foci: list[dict], labels: str, with_permission: bool) -> dict:
    value = _json_object(text)
    keys = {"a", "t", "content"} | ({"permission"} if with_permission else set())
    if set(value) != keys:
        raise ValueError("invalid stage2 keys")
    if value["a"] not in set(LABELS[labels]) | {"unknown"} or not isinstance(value["t"], str):
        raise ValueError("invalid action")
    content = value["content"]
    if not isinstance(content, dict) or set(content) != {f["id"] for f in foci}:
        raise ValueError("invalid content keys")
    if any(v not in CONTENT_VALUES for v in content.values()):
        raise ValueError("invalid content value")
    permission = value["permission"] if with_permission else "ok"
    if permission not in PERMISSION_VALUES:
        raise ValueError("invalid permission")
    return {"a_m": value["a"], "t_m": value["t"], "permission": permission, "content": content}


def combine(foci: list[dict], judged: dict) -> str:
    """D18 §2.3: permission veto first; keep iff some focus has a_m∈A(f) and compatible."""
    if judged["permission"] == "violation":
        return "drop"
    if judged["permission"] == "unknown" or judged["a_m"] == "unknown":
        return "unknown"
    matched = [f for f in foci if judged["a_m"] in f["actions"]]
    if any(judged["content"][f["id"]] == "compatible" for f in matched):
        return "keep"
    if any(judged["content"][f["id"]] == "unknown" for f in matched) or any(
        f["actions"] == ["unknown"] for f in foci
    ):
        return "unknown"
    return "drop"


def parse_card(text: str, candidate: dict, labels: str) -> dict:
    value = _json_object(text)
    if set(value) != {"msgs", "persons", "premises", "needs_permission"}:
        raise ValueError("invalid card keys")
    msgs, premises, needs = value["msgs"], value["premises"], value["needs_permission"]
    if not isinstance(msgs, list) or len(msgs) != len(candidate["r"]):
        raise ValueError("card msgs must cover every reply")
    for i, msg in enumerate(msgs, 1):
        if not isinstance(msg, dict) or set(msg) != {"i", "action", "evidence"}:
            raise ValueError("invalid card msg")
        if msg["i"] != i or msg["action"] not in LABELS[labels] or not isinstance(
            msg["evidence"], str,
        ):
            raise ValueError("invalid card msg value")
    if not _string_list(value["persons"]):
        raise ValueError("invalid persons")
    if not isinstance(premises, list) or not 1 <= len(premises) <= 5:
        raise ValueError("card needs 1-5 premises")
    ids = set()
    for premise in premises:
        if not isinstance(premise, dict) or set(premise) != {"id", "text", "kind"}:
            raise ValueError("invalid premise")
        pid = premise["id"]
        if not isinstance(pid, str) or not pid or pid in ids or not isinstance(
            premise["text"], str,
        ) or premise["kind"] not in PREMISE_KINDS:
            raise ValueError("invalid premise value")
        ids.add(pid)
    if not isinstance(needs, dict) or set(needs) != {"nickname", "tease"} or not all(
        isinstance(v, bool) for v in needs.values()
    ):
        raise ValueError("invalid needs_permission")
    return value


def parse_judge2(text: str, card: dict, with_permission: bool) -> dict:
    value = _json_object(text)
    keys = {"premises", "key_missing"} | ({"permission"} if with_permission else set())
    if set(value) != keys:
        raise ValueError("invalid judge2 keys")
    premises = value["premises"]
    if not isinstance(premises, dict) or set(premises) != {p["id"] for p in card["premises"]}:
        raise ValueError("judge2 must cover every card premise")
    if any(v not in PREMISE_VALUES for v in premises.values()):
        raise ValueError("invalid premise verdict")
    if not isinstance(value["key_missing"], bool):
        raise ValueError("invalid key_missing")
    permission = value["permission"] if with_permission else None
    if with_permission and permission not in PERMISSION_VALUES:
        raise ValueError("invalid permission")
    return {"premises": premises, "permission": permission, "key_missing": value["key_missing"]}


def combine2(foci: list[dict], card: dict, judged: dict) -> str:
    """D20 §3 order: permission veto, contradicted premise, unaccepted action, keep, unknown."""
    if judged["permission"] == "violation":
        return "drop"
    verdicts = judged["premises"].values()
    if "contradicted" in verdicts:
        return "drop"
    accepted = {a for f in foci if f["actions"] != ["unknown"] for a in f["actions"]}
    if any(msg["action"] not in accepted for msg in card["msgs"]):
        return "drop"
    if all(v == "supported" for v in verdicts) and judged["permission"] in ("ok", None):
        return "keep"
    return "unknown" if judged["key_missing"] else "drop"


def card_key(candidate: dict, labels: str, model: str) -> str:
    version = CARD_PROMPT_VERSION + ":" + hashlib.sha256(
        card_system(labels).encode()).hexdigest()
    text = "\n".join(candidate["c"]) + "\n" + "\n".join(candidate["r"]) + version + model
    return hashlib.sha256(text.encode()).hexdigest()


class CardCache:
    """Offline candidate cards, one JSON file per key; a hit never regenerates."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.hits = self.misses = 0

    def get(self, backend, candidate: dict, labels: str) -> tuple[str, dict]:
        key = card_key(candidate, labels, backend.name)
        path = self.directory / f"{key}.json"
        if path.exists():
            self.hits += 1
            return key, parse_card(path.read_text(encoding="utf-8"), candidate, labels)
        self.misses += 1
        card = parse_card(
            backend.generate(card_messages(candidate, labels), MAX_TOKENS["card"]),
            candidate, labels,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
        return key, card


def _row(event, candidate, mode, model, decision, p_yes=None, fields=None, failed=False):
    return {
        "event_id": event["event_id"], "cand_id": candidate["cand_id"], "mode": mode,
        "model": model, "decision": decision, "p_yes": p_yes, "fields": fields or {},
        "technical_failure": failed,
    }


def judge_ga2(event: dict, backend, labels: str, card_cache: CardCache) -> list[dict]:
    model, rows = backend.name, []
    try:
        foci = parse_stage1(
            backend.generate(stage1_ga2_messages(event, labels), MAX_TOKENS["stage1"]), labels,
        )
    except Exception:
        return [_row(event, c, "ga2", model, "drop", fields={"stage": "stage1"}, failed=True)
                for c in event["candidates"]]
    with_permission = event.get("permission") is not None
    for candidate in event["candidates"]:
        fields = {"labels": labels, "foci": foci, "stage": "card"}
        try:
            fields["card_key"], card = card_cache.get(backend, candidate, labels)
            fields.update(card=card, stage="judge2")
            judged = parse_judge2(
                backend.generate(judge2_messages(event, candidate, card, foci),
                                 MAX_TOKENS["judge2"]),
                card, with_permission,
            )
        except Exception:
            rows.append(_row(event, candidate, "ga2", model, "drop", fields=fields, failed=True))
            continue
        del fields["stage"]
        rows.append(_row(event, candidate, "ga2", model, combine2(foci, card, judged),
                         fields={**fields, **judged}))
    return rows


def judge_event(event: dict, mode: str, backend, labels: str = "five",
                card_cache: CardCache | None = None) -> list[dict]:
    """Rows for every candidate; any backend/parse error is a technical failure (drop)."""
    if mode == "ga2":
        return judge_ga2(event, backend, labels, card_cache)
    model = backend.name
    rows = []
    foci = None
    if mode == "ga":
        try:
            foci = parse_stage1(
                backend.generate(stage1_messages(event, labels), MAX_TOKENS["stage1"]), labels,
            )
        except Exception:
            return [_row(event, c, mode, model, "drop", fields={"stage": "stage1"}, failed=True)
                    for c in event["candidates"]]
    for candidate in event["candidates"]:
        try:
            if mode == "direct":
                top, probs = backend.choice_probs(direct_messages(event, candidate), ("是", "否"))
                p_yes = probs["是"]
                if top is None:
                    rows.append(_row(event, candidate, mode, model, "unknown", p_yes,
                                     {"top": None}))
                else:
                    rows.append(_row(event, candidate, mode, model,
                                     "keep" if p_yes >= 0.5 else "drop", p_yes, {"top": top}))
            elif mode == "content_only":
                text = backend.generate(content_messages(event, candidate),
                                        MAX_TOKENS["content_only"]).strip()
                content = CONTENT_WORDS[text]
                decision = {"compatible": "keep", "conflict": "drop"}.get(content, "unknown")
                rows.append(_row(event, candidate, mode, model, decision,
                                 fields={"content": content}))
            else:
                judged = parse_stage2(
                    backend.generate(stage2_messages(event, candidate, foci, labels),
                                     MAX_TOKENS["stage2"]),
                    foci, labels, event.get("permission") is not None,
                )
                rows.append(_row(event, candidate, mode, model, combine(foci, judged),
                                 fields={"labels": labels, "foci": foci, **judged}))
        except Exception:
            rows.append(_row(event, candidate, mode, model, "drop",
                             fields={"stage": "stage2" if mode == "ga" else mode}, failed=True))
    return rows


def summarize(events: list[dict], rows: list[dict]) -> dict:
    """Practice metrics: a scene is correct iff its labelled candidates match keep/no-keep."""
    by_key = {(r["event_id"], r["cand_id"]): r for r in rows}
    groups = defaultdict(list)
    counts = dict.fromkeys(("pos_total", "pos_kept", "neg_total", "neg_kept", "unk_total",
                            "unk_correct", "unk_kept"), 0)
    for event in events:
        correct = True
        for cid, label in event.get("labels", {}).items():
            kept = by_key[event["event_id"], cid]["decision"] == "keep"
            if label["app"] == "yes":
                counts["pos_total"] += 1
                counts["pos_kept"] += kept
                correct &= kept
            elif label["app"] == "no":
                counts["neg_total"] += 1
                counts["neg_kept"] += kept
                correct &= not kept
            elif label["app"] == "unknown":
                unknown = by_key[event["event_id"], cid]["decision"] == "unknown"
                counts["unk_total"] += 1
                counts["unk_correct"] += unknown
                counts["unk_kept"] += kept
                correct &= unknown
        if "group" in event:
            groups[event["group"]].append(correct)
    return {
        "groups_correct": sum(all(v) for v in groups.values()), "groups_total": len(groups),
        **counts,
        "unknown": sum(r["decision"] == "unknown" for r in rows),
        "technical_failures": sum(r["technical_failure"] for r in rows),
        "candidates": len(rows),
        "group_results": {g: all(v) for g, v in sorted(groups.items())},
    }


def model_name(path: str) -> str:
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{8})", path)
    return f"{match[1]}/{match[2]}@{match[3]}" if match else Path(path).name


class MlxBackend:
    """Greedy mlx-lm inference; Qwen3 thinking disabled via the chat template."""

    def __init__(self, path: str):
        import mlx.core as mx
        from mlx_lm import load, stream_generate

        self._mx, self._stream = mx, stream_generate
        self.model, self.tokenizer = load(path)
        self.name = model_name(path)

    def _tokens(self, messages: list[dict]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
        )

    def generate(self, messages: list[dict], max_tokens: int) -> str:
        """Greedy decode; a leading JSON object ends generation and is returned alone.

        Qwen tends to keep writing after the closing brace (often inside the same
        token), so text after the first complete top-level object is cut off.
        """
        text = ""
        for chunk in self._stream(self.model, self.tokenizer, self._tokens(messages),
                                  max_tokens=max_tokens):
            text += chunk.text
            if text.startswith("{") and "}" in chunk.text:
                try:
                    _, end = json.JSONDecoder().raw_decode(text)
                except ValueError:
                    continue
                return text[:end]
        return text

    def choice_probs(self, messages: list[dict], options: tuple[str, ...]):
        mx = self._mx
        ids = []
        for option in options:
            encoded = self.tokenizer.encode(option, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("option is not a single token")
            ids.append(encoded[0])
        logits = self.model(mx.array([self._tokens(messages)]))[0, -1].astype(mx.float32)
        chosen = mx.softmax(logits[mx.array(ids)], axis=-1).tolist()
        top = int(mx.argmax(logits).item())
        probs = dict(zip(options, chosen, strict=True))
        return (options[ids.index(top)] if top in ids else None), probs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True, help="local mlx model snapshot path")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--labels", choices=tuple(LABELS), default="five")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card-cache", type=Path, help="card directory (required for ga2)")
    args = parser.parse_args(argv)
    if args.mode == "ga2" and args.card_cache is None:
        parser.error("--card-cache is required for --mode ga2")
    card_cache = CardCache(args.card_cache) if args.mode == "ga2" else None
    os.environ["HF_HUB_OFFLINE"] = "1"
    events = load_events(args.input)
    started = time.perf_counter()
    try:
        backend = MlxBackend(args.model)
    except Exception:
        backend = None
    load_seconds = time.perf_counter() - started
    rows = []
    started = time.perf_counter()
    for event in events:
        if backend is None:
            rows.extend(_row(event, c, args.mode, model_name(args.model), "drop",
                             fields={"stage": "load"}, failed=True) for c in event["candidates"])
        else:
            rows.extend(judge_event(event, args.mode, backend, args.labels, card_cache))
    seconds = time.perf_counter() - started
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    summary = summarize(events, rows)
    summary.update(mode=args.mode, labels=args.labels, model=model_name(args.model),
                   load_seconds=round(load_seconds, 2),
                   seconds_per_candidate=round(seconds / len(rows), 3) if rows else None,
                   seconds_per_event=round(seconds / len(events), 3) if events else None)
    if card_cache is not None:
        summary.update(card_cache_hits=card_cache.hits, card_cache_misses=card_cache.misses)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
