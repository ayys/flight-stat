"""CLI interface for flight status management."""

import http.server
import socketserver
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from flight_stat import (
    AIRPORTS,
    DB_PATH,
    fetch_flight_status,
    format_airports_list,
    get_airport_codes,
    get_flights_from_db,
    init_database,
    match_airport,
    parse_xml,
    store_flights,
)

console = Console()


def display_flights(flights: List[Dict]) -> None:
    """Display flights in a nice table format.

    Args:
        flights: List of flight dictionaries to display
    """
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


def fetch_all_combinations(conn) -> None:
    """Fetch flight status for all airport combinations.

    Args:
        conn: Database connection
    """
    import time  # Import here to avoid dependency if not needed

    airport_codes = get_airport_codes()

    if not airport_codes:
        console.print("[yellow]No airports found.[/yellow]")
        return

    total_combinations = len(airport_codes) * (len(airport_codes) - 1)
    console.print(
        f"[cyan]Fetching flight status for {total_combinations} route combinations...[/cyan]\n"
    )

    total_flights = 0
    successful_routes = 0
    failed_routes = 0
    route_count = 0

    for departure in airport_codes:
        for arrival in airport_codes:
            if departure == arrival:
                continue  # Skip same airport

            route_count += 1
            try:
                status_msg = (
                    f"[dim][{route_count}/{total_combinations}][/dim] {departure} → {arrival}: "
                )
                xml_content = fetch_flight_status(departure, arrival, verbose=False)

                # Check if response is valid XML (not empty or error)
                if not xml_content or xml_content.strip() == "":
                    console.print(f"{status_msg}[yellow]No data[/yellow]")
                    failed_routes += 1
                    time.sleep(0.1)
                    continue

                # Check for JSON error responses
                if xml_content.strip().startswith("{"):
                    console.print(f"{status_msg}[yellow]API error[/yellow]")
                    failed_routes += 1
                    time.sleep(0.1)
                    continue

                # Parse XML
                try:
                    flights = parse_xml(xml_content)
                except Exception:
                    console.print(f"{status_msg}[yellow]Invalid XML[/yellow]")
                    failed_routes += 1
                    time.sleep(0.1)
                    continue

                if flights:
                    # Store flights
                    inserted, skipped = store_flights(conn, flights)
                    total_flights += inserted
                    successful_routes += 1
                    if inserted > 0:
                        status_parts = [f"{inserted} new"]
                        if skipped > 0:
                            status_parts.append(f"{skipped} skipped")
                        console.print(f"{status_msg}[green]✓ {', '.join(status_parts)}[/green]")
                    else:
                        console.print(f"{status_msg}[dim]No new data ({skipped} skipped)[/dim]")
                else:
                    console.print(f"{status_msg}[dim]No flights[/dim]")
                    failed_routes += 1

                # Small delay to be respectful to the API
                time.sleep(0.1)

            except Exception as e:
                msg = (
                    f"[dim][{route_count}/{total_combinations}][/dim] "
                    f"{departure} → {arrival}: [red]✗ Error: {e}[/red]"
                )
                console.print(msg)
                failed_routes += 1
                time.sleep(0.1)
                continue

    console.print("\n[bold green]Summary:[/bold green]")
    console.print(f"  [green]Successful routes: {successful_routes}[/green]")
    console.print(f"  [yellow]Failed/Empty routes: {failed_routes}[/yellow]")
    console.print(f"  [cyan]Total flights processed: {total_flights}[/cyan]")


def show_db_flights(
    conn, departure_code: Optional[str] = None, arrival_code: Optional[str] = None
) -> None:
    """Show flights from database.

    Args:
        conn: Database connection
        departure_code: Optional departure airport code
        arrival_code: Optional arrival airport code
    """
    flights = get_flights_from_db(conn, departure_code, arrival_code)

    if not flights:
        console.print("[yellow]No flights found in database.[/yellow]")
        if departure_code or arrival_code:
            console.print(
                "[dim]Try fetching flights first with: flight-stat update <FROM> <TO>[/dim]"
            )
        else:
            console.print(
                "[dim]Try fetching flights first with: "
                "flight-stat update <FROM> <TO> or --all[/dim]"
            )
        return

    if departure_code or arrival_code:
        route_str = ""
        if departure_code:
            route_str += f"{AIRPORTS.get(departure_code, departure_code)} ({departure_code})"
        if departure_code and arrival_code:
            route_str += " → "
        if arrival_code:
            route_str += f"{AIRPORTS.get(arrival_code, arrival_code)} ({arrival_code})"
        console.print("[bold green]Flights from Database[/bold green]")
        console.print(f"[cyan]Route: {route_str}[/cyan]\n")
    else:
        console.print("[bold green]Flights from Database[/bold green]")
        console.print("[cyan]Showing all flights[/cyan]\n")

    console.print(f"[green]Found {len(flights)} flight(s)[/green]\n")
    display_flights(flights)


def show_usage():
    """Show usage information."""
    console.print("[red]Error: Invalid usage[/red]\n")
    console.print("[yellow]Usage:[/yellow]")
    console.print("  [cyan]flight-stat show[/cyan]")
    console.print("    Show all flights from database\n")
    console.print("  [cyan]flight-stat show --all[/cyan]")
    console.print("    Show all flights from database\n")
    console.print("  [cyan]flight-stat show <TO> <FROM>[/cyan]")
    console.print("    Show flights for a specific route")
    console.print("    Accepts airport codes, names, or partial matches\n")
    console.print("  [cyan]flight-stat update --all[/cyan]")
    console.print("    Fetch flight status for all route combinations\n")
    console.print("  [cyan]flight-stat update <FROM> <TO>[/cyan]")
    console.print("    Fetch flight status for a specific route")
    console.print("    Accepts airport codes, names, or partial matches\n")
    console.print("  [cyan]flight-stat generate [--output-dir OUTPUT_DIR][/cyan]")
    console.print("    Generate static HTML pages from database\n")
    console.print("  [cyan]flight-stat serve [--output-dir OUTPUT_DIR] [--port PORT][/cyan]")
    console.print("    Serve static HTML pages via local web server")
    console.print("    Default port: 8000\n")
    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")
    console.print("\n[dim]Examples:[/dim]")
    console.print("[dim]  flight-stat show[/dim]")
    console.print("[dim]  flight-stat show --all[/dim]")
    console.print("[dim]  flight-stat show BHR KTM  (TO before FROM)[/dim]")
    console.print("[dim]  flight-stat update KTM BHR[/dim]")
    console.print("[dim]  flight-stat update --all[/dim]")
    console.print("[dim]  flight-stat generate[/dim]")
    console.print("[dim]  flight-stat serve[/dim]")
    console.print("[dim]  flight-stat serve --port 8080[/dim]")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        # No arguments - show usage
        show_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "show":
        # Show flights from database
        conn = init_database()

        try:
            departure = None
            arrival = None

            if len(sys.argv) == 3 and sys.argv[2] == "--all":
                # show --all (show all flights)
                pass
            elif len(sys.argv) == 4:
                # show TO FROM (note: TO before FROM as requested)
                arrival_input = sys.argv[2]  # TO comes first
                departure_input = sys.argv[3]  # FROM comes second

                # Match airport codes/names
                arrival, arr_matches = match_airport(arrival_input)
                if arrival is None:
                    console.print(
                        f"[red]Error: Invalid or ambiguous arrival airport: {arrival_input}[/red]"
                    )
                    if arr_matches:
                        matches_str = ", ".join([f"{name} ({code})" for code, name in arr_matches])
                        console.print(f"[yellow]Ambiguous matches: {matches_str}[/yellow]")
                    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")
                    sys.exit(1)

                departure, dep_matches = match_airport(departure_input)
                if departure is None:
                    console.print(
                        f"[red]Error: Invalid or ambiguous departure airport: "
                        f"{departure_input}[/red]"
                    )
                    if dep_matches:
                        matches_str = ", ".join([f"{name} ({code})" for code, name in dep_matches])
                        console.print(f"[yellow]Ambiguous matches: {matches_str}[/yellow]")
                    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")
                    sys.exit(1)

            show_db_flights(conn, departure_code=departure, arrival_code=arrival)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
        finally:
            conn.close()

    elif command == "update":
        # Update/fetch flights
        conn = init_database()
        console.print(f"[green]Database initialized: {DB_PATH}[/green]\n")

        try:
            if len(sys.argv) == 3 and sys.argv[2] == "--all":
                # update --all (fetch all combinations)
                console.print(
                    "[bold green]Buddha Air - Fetch All Route Combinations[/bold green]\n"
                )
                fetch_all_combinations(conn)
            elif len(sys.argv) == 4:
                # update FROM TO (fetch specific route)
                departure_input = sys.argv[2]
                arrival_input = sys.argv[3]

                # Match airport codes/names
                departure, dep_matches = match_airport(departure_input)
                if departure is None:
                    console.print(
                        f"[red]Error: Invalid or ambiguous departure airport: "
                        f"{departure_input}[/red]"
                    )
                    if dep_matches:
                        matches_str = ", ".join([f"{name} ({code})" for code, name in dep_matches])
                        console.print(f"[yellow]Ambiguous matches: {matches_str}[/yellow]")
                    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")
                    sys.exit(1)

                arrival, arr_matches = match_airport(arrival_input)
                if arrival is None:
                    console.print(
                        f"[red]Error: Invalid or ambiguous arrival airport: {arrival_input}[/red]"
                    )
                    if arr_matches:
                        matches_str = ", ".join([f"{name} ({code})" for code, name in arr_matches])
                        console.print(f"[yellow]Ambiguous matches: {matches_str}[/yellow]")
                    console.print(f"[cyan]Available airports: {format_airports_list()}[/cyan]")
                    sys.exit(1)

                if departure == arrival:
                    airport_name = AIRPORTS[departure]
                    console.print(
                        f"[red]Error: Departure and arrival cannot be the same: "
                        f"{airport_name} ({departure})[/red]"
                    )
                    sys.exit(1)

                # Fetch flight status for specified route
                console.print("[bold green]Buddha Air Flight Status Fetcher[/bold green]")
                dep_name = AIRPORTS[departure]
                arr_name = AIRPORTS[arrival]
                console.print(
                    f"[cyan]Route: {dep_name} ({departure}) → {arr_name} ({arrival})[/cyan]\n"
                )

                xml_content = fetch_flight_status(
                    departure, arrival, verbose=True, progress_callback=console.print
                )
                flights = parse_xml(xml_content)
                console.print(f"[green]Found {len(flights)} flight(s)[/green]\n")

                display_flights(flights)
                inserted, skipped = store_flights(conn, flights)

                status_parts = []
                if inserted > 0:
                    status_parts.append(f"{inserted} new")
                if skipped > 0:
                    status_parts.append(f"{skipped} skipped")

                if inserted > 0:
                    console.print(f"\n[green]✓ {', '.join(status_parts)}[/green]")
                else:
                    console.print(f"\n[yellow]No new data ({skipped} skipped)[/yellow]")
            else:
                console.print("[red]Error: 'update' requires arguments[/red]")
                console.print("[yellow]Usage:[/yellow]")
                console.print("  [cyan]flight-stat update --all[/cyan]")
                console.print("  [cyan]flight-stat update <FROM> <TO>[/cyan]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
        finally:
            conn.close()

    elif command == "generate":
        # Generate static HTML pages
        from flight_stat.generate import generate_pages

        output_dir = "output"
        if len(sys.argv) >= 4 and sys.argv[2] == "--output-dir":
            output_dir = sys.argv[3]

        try:
            conn = init_database()
            console.print(f"[green]Generating static pages in {output_dir}/...[/green]\n")
            generate_pages(conn, output_dir)
            console.print("\n[bold green]✓ Static pages generated successfully![/bold green]")
            console.print(f"[cyan]Open {output_dir}/index.html in your browser[/cyan]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
        finally:
            conn.close()

    elif command == "serve":
        # Serve static HTML pages
        output_dir = "output"
        port = 8000

        # Parse arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output-dir" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    console.print(f"[red]Error: Invalid port number: {sys.argv[i + 1]}[/red]")
                    sys.exit(1)
            elif sys.argv[i] == "--port" and i + 1 >= len(sys.argv):
                console.print("[red]Error: --port requires a port number[/red]")
                sys.exit(1)
            else:
                i += 1

        output_path = Path(output_dir).resolve()
        if not output_path.exists():
            console.print(f"[red]Error: Output directory '{output_dir}' does not exist[/red]")
            console.print(
                "[yellow]Hint: Run 'flight-stat generate' first to generate the pages[/yellow]"
            )
            sys.exit(1)

        if not (output_path / "index.html").exists():
            console.print(f"[red]Error: index.html not found in '{output_dir}'[/red]")
            console.print(
                "[yellow]Hint: Run 'flight-stat generate' first to generate the pages[/yellow]"
            )
            sys.exit(1)

        # Custom handler that serves from the specified directory
        class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(output_path), **kwargs)

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        # Find available port if specified port is in use
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
            console.print(
                f"[red]Error: Could not find an available port starting from {port}[/red]"
            )
            sys.exit(1)

        if actual_port != port:
            console.print(
                f"[yellow]Port {port} is in use, using port {actual_port} instead[/yellow]"
            )

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

    else:
        show_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
