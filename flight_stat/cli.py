"""CLI interface for flight status management."""

import asyncio
import functools
import http.server
import socketserver
from pathlib import Path
from typing import Dict, List, Optional

import click
import httpx
from rich.console import Console
from rich.table import Table

from flight_stat import (
    AIRPORTS,
    analyze_existing_routes,
    fetch_all_combinations_async,
    fetch_flight_status_async,
    format_airports_list,
    get_flights_from_db,
    init_database,
    match_airport,
    parse_xml,
    store_flights,
)

console = Console()


def display_flights(flights: List[Dict]) -> None:
    """Display flights in a nice table format."""
    table = Table(title="Flight Status", show_header=True, header_style="bold magenta")
    table.add_column("Flight No.", style="cyan")
    table.add_column("Departure", style="green")
    table.add_column("Arrival", style="yellow")
    table.add_column("Flight Time", style="blue")
    table.add_column("Revised Time", style="blue")
    table.add_column("Status", style="white")
    table.add_column("Remarks", style="dim")

    for flight in flights:
        table.add_row(
            flight["flight_no"],
            flight["departure"],
            flight["arrival"],
            flight["flight_time"] or "",
            flight["revised_time"] or "",
            flight["flight_status"] or "",
            flight["flight_remarks"] or "",
        )
    console.print(table)


def print_airport_error(input_str: str, matches: Optional[List], airport_type: str) -> None:
    """Print airport matching error."""
    console.print(f"[red]Error: Invalid or ambiguous {airport_type} airport: {input_str}[/red]")
    if matches:
        matches_str = ", ".join([f"{name} ({code})" for code, name in matches])
        console.print(f"[yellow]Ambiguous matches: {matches_str}[/yellow]")
    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")


def async_cmd(f):
    """Decorator to run async commands."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.group()
@click.version_option()
def cli():
    """Flight status management CLI."""
    pass


@cli.command()
@click.argument("to_airport", required=False)
@click.argument("from_airport", required=False)
@click.option("--all", "show_all", is_flag=True, help="Show all flights")
@async_cmd
async def show(to_airport: Optional[str], from_airport: Optional[str], show_all: bool):
    """Show flights from database.

    Examples:
        flight-stat show
        flight-stat show --all
        flight-stat show BHR KTM
    """
    pool = await init_database()

    departure = None
    arrival = None

    if show_all or (not to_airport and not from_airport):
        # show all flights
        pass
    elif to_airport and from_airport:
        # show TO FROM (note: TO before FROM as requested)
        arrival, arr_matches = match_airport(to_airport)
        if arrival is None:
            print_airport_error(to_airport, arr_matches, "arrival")
            raise click.Abort()

        departure, dep_matches = match_airport(from_airport)
        if departure is None:
            print_airport_error(from_airport, dep_matches, "departure")
            raise click.Abort()
    else:
        console.print("[red]Error: Both TO and FROM airports must be provided, or use --all[/red]")
        raise click.Abort()

    flights = await get_flights_from_db(pool, departure, arrival)

    if not flights:
        console.print("[yellow]No flights found in database.[/yellow]")
        console.print("[dim]Try fetching flights first with: flight-stat update <FROM> <TO>[/dim]")
        return

    if departure or arrival:
        route_str = ""
        if departure:
            route_str += f"{AIRPORTS.get(departure, departure)} ({departure})"
        if departure and arrival:
            route_str += " → "
        if arrival:
            route_str += f"{AIRPORTS.get(arrival, arrival)} ({arrival})"
        console.print("[bold green]Flights from Database[/bold green]")
        console.print(f"[cyan]Route: {route_str}[/cyan]\n")
    else:
        console.print("[bold green]Flights from Database[/bold green]")
        console.print("[cyan]Showing all flights[/cyan]\n")

    console.print(f"[green]Found {len(flights)} flight(s)[/green]\n")
    display_flights(flights)


@cli.command()
@click.option("--all", "update_all_flag", is_flag=True, help="Fetch all route combinations")
@click.argument("from_airport", required=False)
@click.argument("to_airport", required=False)
@async_cmd
async def update(update_all_flag: bool, from_airport: Optional[str], to_airport: Optional[str]):
    """Update/fetch flight status.

    Examples:
        flight-stat update --all
        flight-stat update KTM BHR
    """
    if update_all_flag:
        # Fetch all routes
        console.print("[bold green]Buddha Air - Fetch All Route Combinations[/bold green]\n")

        console.print("[cyan]Step 1: Initializing database connection...[/cyan]")
        pool = await init_database()
        console.print("[green]✓ Database connected[/green]\n")

        console.print("[cyan]Step 2: Analyzing existing routes in database...[/cyan]")
        unfetched_count = await analyze_existing_routes(pool, progress_callback=console.print)
        if unfetched_count > 0:
            console.print(f"[cyan]✓ Found {unfetched_count} routes not yet in database (will be checked)[/cyan]\n")
        else:
            console.print("[green]✓ All possible routes have been checked at least once[/green]\n")

        console.print("[cyan]Step 3: Fetching flight status for all route combinations...[/cyan]")
        successful_routes, failed_routes, total_flights = await fetch_all_combinations_async(pool, max_concurrent=20, progress_callback=console.print)

        console.print("\n[bold green]Summary:[/bold green]")
        console.print(f"  [green]Successful routes: {successful_routes}[/green]")
        console.print(f"  [yellow]Failed/Empty routes: {failed_routes}[/yellow]")
        console.print(f"  [cyan]Total flights processed: {total_flights}[/cyan]")
        return

    # Fetch specific route
    if not from_airport or not to_airport:
        console.print("[red]Error: Both FROM and TO airports must be provided, or use --all[/red]")
        raise click.Abort()

    console.print("[bold green]Buddha Air Flight Status Fetcher[/bold green]\n")

    console.print("[cyan]Step 1: Initializing database connection...[/cyan]")
    pool = await init_database()
    console.print("[green]✓ Database connected[/green]\n")

    console.print("[cyan]Step 2: Matching airport codes...[/cyan]")
    console.print(f"  [dim]Departure input: {from_airport}[/dim]")
    departure, dep_matches = match_airport(from_airport)
    if departure is None:
        print_airport_error(from_airport, dep_matches, "departure")
        raise click.Abort()
    console.print(f"  [green]✓ Departure: {AIRPORTS[departure]} ({departure})[/green]")

    console.print(f"  [dim]Arrival input: {to_airport}[/dim]")
    arrival, arr_matches = match_airport(to_airport)
    if arrival is None:
        print_airport_error(to_airport, arr_matches, "arrival")
        raise click.Abort()
    console.print(f"  [green]✓ Arrival: {AIRPORTS[arrival]} ({arrival})[/green]\n")

    if departure == arrival:
        airport_name = AIRPORTS[departure]
        console.print(f"[red]Error: Departure and arrival cannot be the same: {airport_name} ({departure})[/red]")
        raise click.Abort()

    dep_name = AIRPORTS[departure]
    arr_name = AIRPORTS[arrival]
    console.print("[cyan]Step 3: Fetching flight status...[/cyan]")
    console.print(f"  [dim]Route: {dep_name} ({departure}) → {arr_name} ({arrival})[/dim]")

    async with httpx.AsyncClient() as client:
        xml_content = await fetch_flight_status_async(client, departure, arrival)
    console.print("[green]✓ API response received[/green]\n")

    console.print("[cyan]Step 4: Parsing XML response...[/cyan]")
    flights = parse_xml(xml_content)
    console.print(f"[green]✓ Parsed {len(flights)} flight(s)[/green]\n")

    if flights:
        display_flights(flights)
        console.print("\n[cyan]Step 5: Storing flights in database...[/cyan]")
        inserted, skipped = await store_flights(pool, flights, progress_callback=console.print)
        console.print("[green]✓ Database operation completed[/green]")

        status_parts = []
        if inserted > 0:
            status_parts.append(f"{inserted} new")
        if skipped > 0:
            status_parts.append(f"{skipped} skipped")

        if inserted > 0:
            console.print(f"\n[bold green]✓ {', '.join(status_parts)}[/bold green]")
        else:
            console.print(f"\n[yellow]No new data ({skipped} skipped)[/yellow]")
    else:
        console.print("[yellow]No flights found in response[/yellow]")


@cli.command()
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for generated HTML pages",
)
@async_cmd
async def generate(output_dir: str):
    """Generate static HTML pages from database."""
    from flight_stat.generate import generate_pages

    pool = await init_database()
    console.print(f"[green]Generating static pages in {output_dir}/...[/green]\n")
    await generate_pages(pool, output_dir)
    console.print("\n[bold green]✓ Static pages generated successfully![/bold green]")
    console.print(f"[cyan]Open {output_dir}/index.html in your browser[/cyan]")


@cli.command()
@click.option(
    "--output-dir",
    default="output",
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
    help="Directory containing generated HTML pages",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port number to serve on",
)
def serve(output_dir: str, port: int):
    """Serve static HTML pages via local web server."""
    output_path = Path(output_dir).resolve()

    if not (output_path / "index.html").exists():
        console.print(f"[red]Error: index.html not found in '{output_dir}'[/red]")
        console.print("[yellow]Hint: Run 'flight-stat generate' first[/yellow]")
        raise click.Abort()

    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output_path), **kwargs)

        def log_message(self, format, *args):
            pass

    def find_free_port(start_port):
        for test_port in range(start_port, start_port + 100):
            try:
                with socketserver.TCPServer(("", test_port), CustomHTTPRequestHandler):
                    return test_port
            except OSError:
                continue
        return None

    actual_port = find_free_port(port)
    if actual_port is None:
        console.print(f"[red]Error: Could not find an available port starting from {port}[/red]")
        raise click.Abort()

    if actual_port != port:
        console.print(f"[yellow]Port {port} is in use, using port {actual_port} instead[/yellow]")

    httpd = socketserver.TCPServer(("", actual_port), CustomHTTPRequestHandler)
    try:
        console.print(f"[bold green]Serving static pages from {output_path}[/bold green]")
        console.print(f"[cyan]Server running at: http://localhost:{actual_port}[/cyan]")
        console.print("[cyan]Press Ctrl+C to stop[/cyan]\n")
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        httpd.shutdown()
        console.print("[green]Server stopped[/green]")


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
