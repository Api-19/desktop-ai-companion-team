import json

from companion.privacy.ambiguous_models import AmbiguousReviewPackage

_CSP = (
    "default-src 'none'; connect-src 'none'; img-src 'none'; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline';"
)


def _script_safe_json(package: AmbiguousReviewPackage) -> str:
    payload = json.dumps(
        package.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        .replace("</script", "\\u003c/script")
    )


def render_ambiguous_review_html(package: AmbiguousReviewPackage) -> str:
    package_json = _script_safe_json(package)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="{_CSP}">
  <title>Ambiguous Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
    main {{ max-width: 920px; margin: 0 auto; }}
    article {{ border: 1px solid #dadce0; border-radius: 8px; padding: 16px; margin: 12px 0; }}
    .context {{ color: #5f6368; font-size: 14px; margin: 4px 0; }}
    .candidate {{ font-size: 18px; margin: 12px 0; white-space: pre-wrap; }}
    .meta {{ color: #5f6368; font-size: 13px; }}
    .selected {{ outline: 2px solid #1a73e8; }}
    button {{ margin-right: 8px; }}
    button.active {{ font-weight: 700; }}
  </style>
</head>
<body>
  <main>
    <h1>Ambiguous Review</h1>
    <p id="status"></p>
    <p id="context-label">Context is for review only and is not learned.</p>
    <button id="previous">Previous</button>
    <button id="next">Next</button>
    <button id="clear">Clear progress</button>
    <section id="items"></section>
    <button id="export" disabled>Export decisions</button>
    <textarea id="output" readonly rows="8" cols="80"></textarea>
  </main>
  <script>
    const PACKAGE = {package_json};
    const STORAGE_KEY = "ambiguous-review:" + PACKAGE.package_token;
    const decisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}").decisions || {{}};
    let selectedIndex = 0;

    function node(tag, className, text) {{
      const element = document.createElement(tag);
      if (className) element.className = className;
      element.textContent = text;
      return element;
    }}

    function save() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify({{
        packageToken: PACKAGE.package_token,
        decisions
      }}));
    }}

    function decidedCount() {{
      return PACKAGE.items.filter((item) => decisions[item.review_item_id]).length;
    }}

    function exportDocument() {{
      return {{
        package_token: PACKAGE.package_token,
        content_fingerprint: PACKAGE.content_fingerprint,
        decisions: PACKAGE.items.map((item) => ({{
          review_item_id: item.review_item_id,
          decision: decisions[item.review_item_id]
        }}))
      }};
    }}

    function render() {{
      const container = document.getElementById("items");
      container.replaceChildren();
      PACKAGE.items.forEach((item, index) => {{
        const article = document.createElement("article");
        if (index === selectedIndex) article.className = "selected";
        article.appendChild(node("div", "meta", "occurrence_count: " + item.occurrence_count));
        item.context_before.forEach((context) => {{
          article.appendChild(node("div", "context", context.role + ": " + context.text));
        }});
        article.appendChild(node("div", "candidate", item.text));
        item.context_after.forEach((context) => {{
          article.appendChild(node("div", "context", context.role + ": " + context.text));
        }});
        ["keep", "discard", "uncertain"].forEach((value) => {{
          const className = decisions[item.review_item_id] === value ? "active" : "";
          const button = node("button", className, value);
          button.addEventListener("click", () => {{
            decisions[item.review_item_id] = value;
            save();
            render();
          }});
          article.appendChild(button);
        }});
        container.appendChild(article);
      }});
      const count = decidedCount();
      document.getElementById("status").textContent = count + "/" + PACKAGE.items.length;
      document.getElementById("export").disabled = count !== PACKAGE.items.length;
    }}

    function previousItem() {{
      selectedIndex = Math.max(0, selectedIndex - 1);
      render();
    }}

    function nextItem() {{
      selectedIndex = Math.min(PACKAGE.items.length - 1, selectedIndex + 1);
      render();
    }}

    function clearProgress() {{
      Object.keys(decisions).forEach((key) => delete decisions[key]);
      localStorage.removeItem(STORAGE_KEY);
      render();
    }}

    document.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowDown") nextItem();
      if (event.key === "ArrowUp") previousItem();
      if (["1", "2", "3"].includes(event.key) && PACKAGE.items[selectedIndex]) {{
        const value = ["keep", "discard", "uncertain"][Number(event.key) - 1];
        decisions[PACKAGE.items[selectedIndex].review_item_id] = value;
        save();
        render();
      }}
    }});

    document.getElementById("previous").addEventListener("click", previousItem);
    document.getElementById("next").addEventListener("click", nextItem);
    document.getElementById("clear").addEventListener("click", clearProgress);

    document.getElementById("export").addEventListener("click", () => {{
      document.getElementById("output").value = JSON.stringify(exportDocument(), null, 2);
    }});

    render();
  </script>
</body>
</html>
"""
