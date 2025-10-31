import tkinter as tk
from tkinter import messagebox
import threading
import os
from pathlib import Path

from core_finder import index_all, search_docs

# Optional Markdown → HTML and HTML renderer
CAN_RENDER_MARKDOWN = False
_md = None
_HTMLWidgetClass = None

try:
    import markdown as _markdown
    _md = _markdown
    try:
        from tkhtmlview import HTMLScrolledText as _HTMLWidgetClass
        CAN_RENDER_MARKDOWN = True
    except Exception:
        # fallback try tkinterweb
        try:
            from tkinterweb import HtmlFrame as _HTMLWidgetClass
            CAN_RENDER_MARKDOWN = True
        except Exception:
            CAN_RENDER_MARKDOWN = False
except Exception:
    CAN_RENDER_MARKDOWN = False

CONFIG_PATH = "config.yml"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Document Finder + DeepSeek Reasoning")
        self.geometry("980x800")

        tk.Label(self, text="Advanced Document Finder + DeepSeek Reasoning",
                 font=("Segoe UI", 14)).pack(pady=6)

        # Log box
        self.log_box = tk.Text(self, height=6, width=120)
        self.log_box.pack(padx=10, pady=6)
        self._log(f"Markdown render support: {'ON' if CAN_RENDER_MARKDOWN else 'OFF'}")

        # Buttons
        btn_frm = tk.Frame(self)
        btn_frm.pack(pady=4)
        tk.Button(btn_frm, text="Index Documents", width=18,
                  command=self._on_index).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frm, text="Search + Reason", width=18,
                  command=self._on_search).pack(side=tk.LEFT, padx=6)

        # Query
        tk.Label(self, text="Search Query:").pack(anchor="w", padx=12)
        self.query_box = tk.Text(self, height=3, width=120)
        self.query_box.pack(padx=10, pady=6)

        # Result panel
        self.result_container = tk.Frame(self)
        self.result_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        if CAN_RENDER_MARKDOWN and _HTMLWidgetClass is not None and _md is not None:
            # HTML-capable widget
            if _HTMLWidgetClass.__name__ == "HtmlFrame":
                # tkinterweb HtmlFrame has load_html()
                self.result_widget = _HTMLWidgetClass(self.result_container, horizontal_scrollbar="auto")
                self.result_widget.pack(fill=tk.BOTH, expand=True)
                self._render_md("# Ready.\n")
            else:
                # tkhtmlview HTMLScrolledText has set_html()
                self.result_widget = _HTMLWidgetClass(self.result_container, html="<p>Ready.</p>")
                self.result_widget.pack(fill=tk.BOTH, expand=True)
        else:
            # Plain text fallback
            self.result_widget = tk.Text(self.result_container, height=32, width=120)
            self.result_widget.pack(fill=tk.BOTH, expand=True)
            self.result_widget.insert(tk.END, "Ready.\n")

    def _log(self, s: str):
        self.log_box.insert(tk.END, s + "\n")
        self.log_box.see(tk.END)

    def _render_md(self, md_text: str):
        if not CAN_RENDER_MARKDOWN or _md is None or _HTMLWidgetClass is None:
            # fallback: plain text
            self.result_widget.delete("1.0", tk.END)
            self.result_widget.insert(tk.END, md_text)
            return

        html = _md.markdown(md_text, extensions=["fenced_code", "tables", "toc"])
        # Render based on widget type
        if _HTMLWidgetClass.__name__ == "HtmlFrame":
            # tkinterweb
            self.result_widget.load_html(html)
        else:
            # tkhtmlview
            try:
                self.result_widget.set_html(html)
            except Exception:
                # rebuild widget if set_html is missing
                for w in self.result_container.winfo_children():
                    w.destroy()
                from tkhtmlview import HTMLScrolledText
                self.result_widget = HTMLScrolledText(self.result_container, html=html)
                self.result_widget.pack(fill=tk.BOTH, expand=True)

    # ------------- Actions -------------
    def _on_index(self):
        self._log("Indexing started...")
        def task():
            try:
                index_all(CONFIG_PATH)
                self._log("Indexing complete.")
            except Exception as e:
                self._log(f"Error: {e}")
        threading.Thread(target=task, daemon=True).start()

    def _on_search(self):
        q = self.query_box.get("1.0", tk.END).strip()
        if not q:
            messagebox.showwarning("Input", "Please enter a query.")
            return
        self._log(f"Searching: {q}")
        self._render_md("Searching and reasoning...")

        def task():
            try:
                res = search_docs(CONFIG_PATH, q, CAN_RENDER_MARKDOWN)
                # Build final markdown output: answer + matched list
                answer = res.get("answer_md", "")
                matched = res.get("matched", [])
                if matched:
                    files_md = "\n".join([f"- `{m}`" for m in matched])
                else:
                    files_md = "_No matched files above threshold._"
                md_out = f"{answer}\n\n---\n**Matched Files (cosine > 0.3):**\n\n{files_md}"
                self._render_md(md_out)
            except Exception as e:
                self._render_md(f"Error: {e}")

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    if not os.path.exists(CONFIG_PATH):
        messagebox.showerror("Error", f"{CONFIG_PATH} not found.")
        raise SystemExit
    App().mainloop()
