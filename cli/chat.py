import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from app.agents.graph import agent
from app.services.memory import memory_service

logging.basicConfig(level=logging.WARNING)

console = Console()


class LifeInsuranceCLI:
    def __init__(self):
        self.session_id = None
        self.console = Console()

    def display_welcome(self):
        welcome_text = """
        # Life Insurance Support Assistant
        
        Welcome! I'm your AI-powered life insurance assistant. I can help you with:
        
        - Understanding different types of life insurance policies
        - Checking eligibility requirements
        - Calculating premium estimates
        - Learning about the claims process
        - Comparing coverage options
        - Answering general life insurance questions
        
        Ask me anything about life insurance, and I'll provide accurate, helpful information.
        """

        self.console.print(
            Panel(
                Markdown(welcome_text),
                title="[bold cyan]Welcome[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def display_help(self):
        table = Table(title="Available Commands", box=box.ROUNDED, border_style="blue")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")

        table.add_row("/help", "Show this help message")
        table.add_row("/clear", "Clear conversation history")
        table.add_row("/history", "Show conversation history")
        table.add_row("/new", "Start a new session")
        table.add_row("/quit or /exit", "Exit the application")

        self.console.print(table)
        self.console.print()

    def display_history(self):
        if not self.session_id:
            self.console.print("[yellow]No active session[/yellow]")
            return

        messages = memory_service.get_conversation_history(self.session_id)

        if not messages:
            self.console.print("[yellow]No conversation history yet[/yellow]")
            return

        self.console.print(
            Panel("[bold]Conversation History[/bold]", border_style="blue")
        )

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                self.console.print(
                    Panel(
                        content,
                        title="[bold blue]You[/bold blue]",
                        border_style="blue",
                        box=box.SIMPLE,
                    )
                )
            else:
                self.console.print(
                    Panel(
                        Markdown(content),
                        title="[bold green]Assistant[/bold green]",
                        border_style="green",
                        box=box.SIMPLE,
                    )
                )

        self.console.print()

    def display_response(self, response: dict):
        answer = response.get("answer", "No response generated")
        sources = response.get("sources", [])
        reasoning = response.get("agent_reasoning", "")

        self.console.print(
            Panel(
                Markdown(answer),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

        if sources:
            sources_text = Text()
            sources_text.append("Sources: ", style="bold cyan")
            sources_text.append(
                ", ".join([s.split("/")[-1] for s in sources]), style="dim"
            )
            self.console.print(sources_text)

        if reasoning:
            self.console.print(Text(f"[{reasoning}]", style="dim italic"))

        self.console.print()

    def create_session(self):
        self.session_id = memory_service.create_session()
        self.console.print(f"[dim]Session created: {self.session_id}[/dim]\n")

    def clear_session(self):
        if self.session_id:
            memory_service.clear_session(self.session_id)
            self.console.print("[yellow]Conversation history cleared[/yellow]")
            self.create_session()
        else:
            self.console.print("[yellow]No active session to clear[/yellow]")

    def process_command(self, user_input: str) -> bool:
        command = user_input.lower().strip()

        if command in ["/quit", "/exit"]:
            self.console.print(
                Panel(
                    "[bold cyan]Thank you for using the Life Insurance Support Assistant!\nGoodbye![/bold cyan]",
                    border_style="cyan",
                )
            )
            return False

        elif command == "/help":
            self.display_help()

        elif command == "/clear":
            self.clear_session()

        elif command == "/history":
            self.display_history()

        elif command == "/new":
            self.clear_session()
            self.console.print("[green]New session started[/green]\n")

        else:
            self.console.print(
                "[yellow]Unknown command. Type /help for available commands.[/yellow]\n"
            )

        return True

    def run(self):
        load_dotenv()

        self.display_welcome()
        self.create_session()

        self.console.print(
            "[dim]Type /help for commands or ask any question about life insurance.[/dim]"
        )
        self.console.print("[dim]Type /quit to exit.[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    should_continue = self.process_command(user_input)
                    if not should_continue:
                        break
                    continue

                with self.console.status(
                    "[bold green]Thinking...[/bold green]", spinner="dots"
                ):
                    response = agent.process_message(
                        message=user_input, session_id=self.session_id
                    )

                memory_service.add_message(
                    session_id=self.session_id, role="user", content=user_input
                )

                memory_service.add_message(
                    session_id=self.session_id,
                    role="assistant",
                    content=response["answer"],
                )

                self.display_response(response)

            except KeyboardInterrupt:
                self.console.print(
                    "\n[yellow]Interrupted. Type /quit to exit properly.[/yellow]"
                )
                continue

            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")
                logging.error(f"CLI Error: {str(e)}", exc_info=True)


def main():
    cli = LifeInsuranceCLI()
    cli.run()


if __name__ == "__main__":
    main()
