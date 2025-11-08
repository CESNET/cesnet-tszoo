import os
from dataclasses import dataclass, field
from IPython.display import HTML, display


def __get_path_to_styles_folder() -> str:

    return os.path.dirname(__file__)


def __get_css_styles(identifier: str) -> str:
    path = os.path.join(__get_path_to_styles_folder(), f"{identifier}.css")

    css_content = ""

    with open(path, "r", encoding="utf-8") as f:
        css_content = f.read()

    return f"<style>{css_content}</style>"


@dataclass
class SummaryDiagramStep:
    title: str
    additional_data: dict[str, object]
    table_rows: str = field(init=False, default_factory=lambda: "")

    def __post_init__(self):
        if self.additional_data is None:
            return

        for key in self.additional_data:

            row = f"""
                    <tr>
                        <td>{str(key)}</td>
                        <td>{", ".join(self.additional_data[key]) if isinstance(self.additional_data[key], list) else self.additional_data[key]}</td>
                    </tr>          
            """
            self.table_rows = self.table_rows + " " + row

    def get_css_body(self) -> str:

        if self.additional_data is None:
            body = f"""
            <div class="pipe-step">
                <details class="dropdown">
                    <summary class="empty-dropbtn">{self.title}</summary>
                </details>
            <div>
            """
        else:

            body = f"""
            <div class="pipe-step">
                <details class="dropdown">
                    <summary class="dropbtn">{self.title}</summary>
                    <div class="dropdown-content">
                        <table>
                            {self.table_rows}                      
                        </table>
                    </div>
                </details>
            <div>
            """

        return body


def display_summary_diagram(steps: list[SummaryDiagramStep]) -> None:
    styles = __get_css_styles("summary_diagram")

    html = [styles, '<div class="pipe-container">', '<div class="pipe-title">Preprocessing sequence</div>']

    for i, step in enumerate(steps):
        html.append(step.get_css_body())

        if i < len(steps) - 1:
            html.append('<div class="pipe-connector"></div>')

    html.append("</div>")
    html = "".join(html)

    display(HTML(html))
