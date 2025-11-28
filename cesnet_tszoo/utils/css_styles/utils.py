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
class StepAttribute:
    name: str
    value: object

    def get_css_body(self) -> str:
        return f"""
                <tr>
                    <td>{self.name}</td>
                    <td>{", ".join(self.value) if isinstance(self.value, list) else str(self.value)}</td>
                </tr>       
                """


@dataclass
class SummaryDiagramStep:
    title: str
    attributes: list[StepAttribute]
    table_rows: str = field(init=False, default_factory=lambda: "")

    def __post_init__(self):
        if self.attributes is None or len(self.attributes) == 0:
            self.attributes = None
            return

        for attribute in self.attributes:
            self.table_rows = self.table_rows + " " + attribute.get_css_body()

    def get_css_body(self) -> str:

        if self.attributes is None:
            body = f"""
            <div class="pipe-step">
                <details class="dropdown">
                    <summary class="empty-dropbtn">{self.title}</summary>
                </details>
            </div>
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
            </div>
            """

        return body


def get_summary_diagram(steps: list[SummaryDiagramStep]) -> str:
    styles = __get_css_styles("summary_diagram")

    fallback_msg = (
        """
    In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. 
    <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
    """
    )

    html = [styles, f'<div class="text-fallback">{fallback_msg}</div>', '<div class="pipe-container" hidden>', '<div class="pipe-title">Preprocessing sequence</div>']

    for i, step in enumerate(steps):
        html.append(step.get_css_body())

        if i < len(steps) - 1:
            html.append('<div class="pipe-connector"></div>')

    html.append("</div>")
    html = "".join(html)

    return html


def display_summary_diagram(steps: list[SummaryDiagramStep]) -> None:

    html = get_summary_diagram(steps)

    display(HTML(html))
