"""Static HTML page generator for flight status data using Jinja2 templates."""

import contextlib
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from flight_stat.lib import (
    AIRPORTS,
    get_flights_for_route,
    get_flights_from_db,
    get_nepal_date,
    get_unique_flight_routes,
)


def get_airport_code_from_name(airport_name: str) -> str:
    """Get airport code from airport name stored in database."""
    airport_name_upper = airport_name.upper()
    # Remove parenthetical info
    airport_name_base = airport_name_upper.split("(")[0].strip()

    for code, name in AIRPORTS.items():
        name_base = name.upper().split("(")[0].strip()
        if name_base == airport_name_base or name_base in airport_name_base or airport_name_base in name_base:
            return code

    # If not found, try to extract from the name itself
    return airport_name[:3].upper()


def sanitize_filename(text: str) -> str:
    """Sanitize text for use in filename."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def normalize_date(d):
    """Normalize date to date object."""
    if isinstance(d, date):
        return d
    elif isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    else:
        return get_nepal_date()


def get_status_badge_class(flight_status: str) -> str:
    """Get Bootstrap badge class based on flight status."""
    if not flight_status:
        return "bg-secondary"

    status_lower = flight_status.lower()
    if "on time" in status_lower or "scheduled" in status_lower:
        return "bg-success"
    elif "delayed" in status_lower:
        return "bg-warning"
    elif "cancelled" in status_lower or "canceled" in status_lower:
        return "bg-danger"
    return "bg-secondary"


def prepare_flight_data(flight: Dict) -> Dict:
    """Prepare flight data for template rendering."""
    dep_code = get_airport_code_from_name(flight["departure"])
    arr_code = get_airport_code_from_name(flight["arrival"])
    flight_date_obj = normalize_date(flight["flight_date"])

    return {
        "flight_no": flight["flight_no"],
        "departure": flight["departure"],
        "arrival": flight["arrival"],
        "route_display": f"{flight['departure']} → {flight['arrival']}",
        "flight_time": flight.get("flight_time", "") or None,
        "revised_time": flight.get("revised_time", "") or None,
        "flight_status": flight.get("flight_status", "") or None,
        "flight_remarks": flight.get("flight_remarks", "") or None,
        "flight_date_str": flight_date_obj.strftime("%Y-%m-%d"),
        "status_badge_class": get_status_badge_class(flight.get("flight_status", "")),
        "filename": (f"flight_{sanitize_filename(dep_code)}_{sanitize_filename(arr_code)}_{sanitize_filename(flight['flight_no'])}.html"),
    }


async def generate_index_page(conn, output_dir: Path, all_flights: List[Dict], env: Environment) -> None:
    """Generate the index page with all flights, filterable by date and route."""

    # Get date range from all flights
    if all_flights:
        dates = sorted({normalize_date(f["flight_date"]) for f in all_flights}, reverse=True)
        min_date = min(dates)
        max_date = max(dates)
        today = get_nepal_date()
        default_date = today if today in dates else dates[0]

        # Get unique airports from flights
        unique_departures_raw = sorted({f["departure"] for f in all_flights})
        unique_arrivals_raw = sorted({f["arrival"] for f in all_flights})
    else:
        min_date = get_nepal_date()
        max_date = get_nepal_date()
        default_date = get_nepal_date()
        unique_departures_raw = []
        unique_arrivals_raw = []

    # Prepare airport data for template
    unique_departures = [{"name": dep, "code": get_airport_code_from_name(dep)} for dep in unique_departures_raw]
    unique_arrivals = [{"name": arr, "code": get_airport_code_from_name(arr)} for arr in unique_arrivals_raw]

    # Prepare flights data for template
    flights_data = [prepare_flight_data(flight) for flight in all_flights]
    flights_data.sort(
        key=lambda x: (
            x["flight_date_str"],
            x["departure"],
            x["arrival"],
            x["flight_no"],
            x["flight_time"] or "",
        )
    )

    template = env.get_template("index.html")
    html_content = template.render(
        default_date=default_date,
        min_date=min_date,
        max_date=max_date,
        unique_departures=unique_departures,
        unique_arrivals=unique_arrivals,
        flights=flights_data,
    )

    index_path = output_dir / "index.html"
    index_path.write_text(html_content, encoding="utf-8")


async def generate_flight_page(conn_or_flights, output_dir: Path, departure: str, arrival: str, flight_no: str, env: Environment) -> None:
    """Generate an individual flight page with historical data.

    Args:
        conn_or_flights: Either a database connection (for backward compatibility)
                        or a list of flight dictionaries (optimized path)
        output_dir: Output directory for HTML files
        departure: Departure airport name
        arrival: Arrival airport name
        flight_no: Flight number
        env: Jinja2 environment
    """
    # Accept either a connection (legacy) or pre-fetched flights list (optimized)
    if isinstance(conn_or_flights, list):
        # Filter flights for this route + flight number
        all_flights = [
            f
            for f in conn_or_flights
            if f.get("departure", "").upper() == departure.upper()
            and f.get("arrival", "").upper() == arrival.upper()
            and f.get("flight_no", "") == flight_no
        ]
    else:
        # Legacy path: fetch from database
        all_flights = await get_flights_for_route(conn_or_flights, departure, arrival, flight_no)

    if not all_flights:
        return

    # Get date range - normalize dates to date objects
    dates = sorted({normalize_date(f["flight_date"]) for f in all_flights}, reverse=True)
    min_date = min(dates)
    max_date = max(dates)
    today = get_nepal_date()
    default_date = today if today in dates else dates[0]

    # Prepare flights data for template
    flights_data = []
    for flight in all_flights:
        flight_date_obj = normalize_date(flight["flight_date"])
        fetched_at_str = ""
        if flight.get("fetched_at"):
            with contextlib.suppress(Exception):
                fetched_at_str = flight["fetched_at"][:16] if isinstance(flight["fetched_at"], str) else str(flight["fetched_at"])[:16]

        flights_data.append(
            {
                "flight_date_str": flight_date_obj.strftime("%Y-%m-%d"),
                "flight_time": flight.get("flight_time", "") or None,
                "revised_time": flight.get("revised_time", "") or None,
                "flight_status": flight.get("flight_status", "") or None,
                "flight_remarks": flight.get("flight_remarks", "") or None,
                "fetched_at_str": fetched_at_str,
                "status_badge_class": get_status_badge_class(flight.get("flight_status", "")),
            }
        )

    dep_code = get_airport_code_from_name(departure)
    arr_code = get_airport_code_from_name(arrival)

    template = env.get_template("flight.html")
    html_content = template.render(
        flight_no=flight_no,
        departure=departure,
        arrival=arrival,
        default_date=default_date,
        min_date=min_date,
        max_date=max_date,
        flights=flights_data,
    )

    filename = f"flight_{sanitize_filename(dep_code)}_{sanitize_filename(arr_code)}_{sanitize_filename(flight_no)}.html"
    flight_path = output_dir / filename
    flight_path.write_text(html_content, encoding="utf-8")


async def generate_pages(conn, output_dir: str = "output") -> None:
    """Generate static HTML pages from database.

    Args:
        conn: Database connection
        output_dir: Output directory for HTML files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup Jinja2 environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape(["html", "xml"]))

    # Get all flights (no date filter for index page - it will filter client-side)
    all_flights = await get_flights_from_db(conn)

    # Generate index page
    await generate_index_page(conn, output_path, all_flights, env)

    # Get unique routes for generating individual flight pages
    routes = await get_unique_flight_routes(conn)

    # Generate individual flight pages
    # Pass all_flights directly to avoid DB calls in the loop
    for route in routes:
        await generate_flight_page(all_flights, output_path, route["departure"], route["arrival"], route["flight_no"], env)
