"""Render the owner-only character name form."""

import html


def render_character_form(*, name: str, csrf: str, token: str) -> str:
    return (
        "<h1>角色名称</h1>"
        "<p>保存后需重启对话才生效。</p>"
        '<form method="post" action="/admin/character">'
        f'<input type="hidden" name="csrf" value="{html.escape(csrf, quote=True)}">'
        '<input type="hidden" name="character_token" '
        f'value="{html.escape(token, quote=True)}">'
        "<label>角色名称 "
        f'<input name="name" value="{html.escape(name, quote=True)}"></label>'
        '<label><input type="checkbox" name="confirm" value="yes">确认保存角色名称</label>'
        "<button>保存</button></form>"
    )
