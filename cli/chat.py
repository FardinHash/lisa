import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
from typing import Dict

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from app.agents.graph import agent
from app.services.memory import memory_service

logging.basicConfig(level=logging.WARNING)

console = Console()


class LifeInsuranceCLI:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.console = Console()

    def display_welcome(self) -> None:
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                    ██╗     ██╗███████╗ █████╗                ║
║                    ██║     ██║██╔════╝██╔══██╗               ║
║                    ██║     ██║███████╗███████║               ║
║                    ██║     ██║╚════██║██╔══██║               ║
║                    ███████╗██║███████║██║  ██║               ║
║                    ╚══════╝╚═╝╚══════╝╚═╝  ╚═╝               ║
║                                                              ║
║                    Life Insurance Support Assistant          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """

        self.console.print(f"[bold cyan]{banner}[/bold cyan]", justify="center")
        self.console.print()

        capabilities_left = Panel(
            "[bold]What I Can Help With:[/bold]\n\n"
            "• Policy Types & Comparison\n"
            "• Eligibility Requirements\n"
            "• Premium Calculations",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )

        capabilities_right = Panel(
            "[bold]Features:[/bold]\n\n"
            "• Claims Process Guide\n"
            "• Coverage Recommendations\n"
            "• Expert Q&A Support",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )

        self.console.print(
            Columns([capabilities_left, capabilities_right]), justify="center"
        )
        self.console.print()

        examples = Panel(
            "[bold yellow]Try asking:[/bold yellow]\n"
            '[dim]"What types of life insurance are available?"[/dim]\n'
            '[dim]"Calculate premium for 35 year old, $500k coverage"[/dim]\n'
            '[dim]"Can I get insurance if I have diabetes?"[/dim]',
            border_style="yellow",
            box=box.SIMPLE,
            padding=(0, 2),
        )

        self.console.print(examples)
        self.console.print()

    def display_help(self) -> None:
        table = Table(
            title="[bold cyan]Available Commands[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Command", style="cyan bold", no_wrap=True, width=20)
        table.add_column("Description", style="white")

        table.add_row("help", "Show this help message")
        table.add_row("clear", "Clear conversation history and start fresh")
        table.add_row("history", "Show full conversation history")
        table.add_row("new", "Start a new session")
        table.add_row("quit or exit", "Exit the application")

        self.console.print()
        self.console.print(table)
        self.console.print()

    def display_history(self) -> None:
        if not self.session_id:
            self.console.print("\n[yellow] No active session[/yellow]\n")
            return

        messages = memory_service.get_conversation_history(self.session_id)

        if not messages:
            self.console.print("\n[yellow] No conversation history yet[/yellow]\n")
            return

        self.console.print()
        self.console.print(
            Rule(
                f"[bold cyan]Conversation History ({len(messages)} messages)[/bold cyan]"
            )
        )
        self.console.print()

        for idx, msg in enumerate(messages, 1):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                self.console.print(
                    Panel(
                        f"[white]{content}[/white]",
                        title=f"[bold blue] You[/bold blue] [dim]#{idx}[/dim]",
                        border_style="blue",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            else:
                self.console.print(
                    Panel(
                        Markdown(content),
                        title=f"[bold green] Assistant[/bold green] [dim]#{idx}[/dim]",
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )

            if idx < len(messages):
                self.console.print()

        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()

    def display_response(self, response: Dict[str, any]) -> None:
        answer = response.get("answer", "No response generated")
        sources = response.get("sources", [])
        reasoning = response.get("agent_reasoning", "")

        self.console.print()
        self.console.print(
            Panel(
                Markdown(answer),
                title="[bold green] Assistant[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

        footer_items = []

        if sources:
            source_names = [s.split("/")[-1].replace(".txt", "") for s in sources]
            sources_text = Text()
            sources_text.append(" ", style="bold cyan")
            sources_text.append("Sources: ", style="bold cyan")
            sources_text.append(", ".join(source_names), style="cyan")
            footer_items.append(sources_text)

        if reasoning:
            reasoning_text = Text()
            reasoning_text.append(" ", style="bold magenta")
            reasoning_text.append(f"{reasoning}", style="dim italic magenta")
            footer_items.append(reasoning_text)

        if footer_items:
            self.console.print()
            for item in footer_items:
                self.console.print(item)

        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()

    def create_session(self) -> None:
        self.session_id = memory_service.create_session()
        session_short = self.session_id[:8]
        self.console.print(f"[dim]✓ Session created: {session_short}...[/dim]\n")

    def clear_session(self) -> None:
        if self.session_id:
            memory_service.clear_session(self.session_id)
            self.console.print("\n[green]✓ Conversation history cleared[/green]")
            self.create_session()
        else:
            self.console.print("\n[yellow]⚠ No active session to clear[/yellow]\n")

    def process_command(self, user_input: str) -> bool:
        command = user_input.lower().strip().lstrip("/")

        if command in ["quit", "exit"]:
            self.console.print()
            goodbye_panel = Panel(
                Align.center(
                    "[bold cyan]Thank you for using the\n"
                    "Life Insurance Support Assistant!\n\n"
                    "👋 Goodbye![/bold cyan]"
                ),
                border_style="cyan",
                box=box.DOUBLE,
            )
            self.console.print(goodbye_panel)
            self.console.print()
            return False

        elif command == "help":
            self.display_help()

        elif command == "clear":
            self.clear_session()

        elif command == "history":
            self.display_history()

        elif command == "new":
            self.clear_session()
            self.console.print("[green]✓ New session started[/green]\n")

        else:
            self.console.print(f"\n[yellow]⚠ Unknown command: {command}[/yellow]")
            self.console.print("[dim]Type 'help' for available commands[/dim]\n")

        return True

    def run(self) -> None:
        load_dotenv()

        self.display_welcome()
        self.create_session()

        info_panel = Panel(
            "[dim]💡 Type [bold cyan]help[/bold cyan] for commands  |  "
            "Ask any question about life insurance  |  "
            "Type [bold cyan]quit[/bold cyan] to exit[/dim]",
            border_style="dim",
            box=box.SIMPLE,
        )
        self.console.print(info_panel)
        self.console.print()

        message_count = 0

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]👤 You[/bold blue]")

                if not user_input.strip():
                    continue

                cleaned_input = user_input.strip().lstrip("/").lower()
                known_commands = ["help", "clear", "history", "new", "quit", "exit"]

                if cleaned_input in known_commands:
                    should_continue = self.process_command(user_input)
                    if not should_continue:
                        break
                    continue

                with self.console.status(
                    "[bold green] Thinking...[/bold green]", spinner="dots12"
                ):
                    start_time = time.time()
                    response = agent.process_message(
                        message=user_input, session_id=self.session_id
                    )
                    elapsed = time.time() - start_time

                memory_service.add_message(
                    session_id=self.session_id, role="user", content=user_input
                )

                memory_service.add_message(
                    session_id=self.session_id,
                    role="assistant",
                    content=response["answer"],
                )

                self.display_response(response)

                message_count += 1
                self.console.print(
                    f"[dim]⏱ Response time: {elapsed:.2f}s  |  "
                    f"Messages in session: {message_count * 2}[/dim]\n"
                )

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]⚠ Interrupted. Type 'quit' to exit properly.[/yellow]\n"
                )
                continue

            except Exception as e:
                self.console.print()
                error_panel = Panel(
                    f"[bold red]Error:[/bold red] {str(e)}\n\n"
                    "[dim]Please try again or contact support if the issue persists.[/dim]",
                    title="[bold red] Error[/bold red]",
                    border_style="red",
                )
                self.console.print(error_panel)
                self.console.print()
                logging.error(f"CLI Error: {str(e)}", exc_info=True)


def main() -> None:
    cli = LifeInsuranceCLI()
    cli.run()


if __name__ == "__main__":
    main()
