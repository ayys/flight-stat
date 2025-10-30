"""Core library functions for flight status management.

This module provides framework-agnostic functions for database operations,
API interactions, and data processing. No Rich console dependencies.
"""

import asyncio
import sqlite3
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import requests
from zoneinfo import ZoneInfo

# Nepal timezone (UTC+5:45)
NEPAL_TZ = ZoneInfo("Asia/Kathmandu")

# Database file path
DB_PATH = Path.home() / "flight_status.db"

# Airport codes and names
AIRPORTS = {
    "KTM": "Kathmandu",
    "MTN": "Everest mountain flight",
    "BDP": "Bhadrapur",
    "BWA": "Bhairahawa",
    "BHR": "Bharatpur (chitwan)",
    "BIR": "Biratnagar",
    "DHI": "Dhangadhi",
    "JKR": "Janakpur",
    "JMO": "Jomsom",
    "LUA": "Lukla",
    "RHP": "Ramechhap",
    "TPU": "Tikapur",
    "CCU": "Kolkata",
    "KEP": "Nepalgunj",
    "PKR": "Pokhara",
    "RJB": "Rajbiraj",
    "SIF": "Simara",
    "SKH": "Surkhet",
    "TMI": "Tumlingtar",
    "VNS": "Varanasi",
}


def init_database():
    """Initialize SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_no TEXT NOT NULL,
            departure TEXT NOT NULL,
            arrival TEXT NOT NULL,
            flight_time TEXT NOT NULL,
            revised_time TEXT NOT NULL,
            flight_status TEXT,
            flight_remarks TEXT,
            flight_date DATE NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add date column if it doesn't exist (for existing databases)
    cursor.execute("""
        SELECT COUNT(*) FROM pragma_table_info('flights') WHERE name='flight_date'
    """)
    if cursor.fetchone()[0] == 0:
        cursor.execute("ALTER TABLE flights ADD COLUMN flight_date DATE")
        # Set default date for existing records
        cursor.execute(
            "UPDATE flights SET flight_date = DATE(fetched_at) WHERE flight_date IS NULL"
        )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS airports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


def fetch_flight_status(
    departure: str,
    arrival: str,
    verbose: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Fetch flight status from the API.

    Args:
        departure: Departure airport code
        arrival: Arrival airport code
        verbose: Whether to print status messages
        progress_callback: Optional callback function for progress messages

    Returns:
        XML content as string
    """
    url = f"https://admin.buddhaair.com/api/flight-status/{departure}/{arrival}"

    if verbose and progress_callback:
        progress_callback(f"Fetching flight status from {url}...")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        if verbose and progress_callback:
            progress_callback(f"Error fetching flight status: {e}")
        raise


async def fetch_flight_status_async(
    client: httpx.AsyncClient,
    departure: str,
    arrival: str,
) -> str:
    """Fetch flight status from the API asynchronously.

    Args:
        client: httpx async client
        departure: Departure airport code
        arrival: Arrival airport code

    Returns:
        XML content as string

    Raises:
        httpx.HTTPError: If the request fails
    """
    url = f"https://admin.buddhaair.com/api/flight-status/{departure}/{arrival}"

    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError:
        raise


def get_airport_codes() -> List[str]:
    """Get all airport codes from the AIRPORTS dictionary."""
    return sorted(AIRPORTS.keys())


def match_airport(input_str: str) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]]]:
    """Match airport input (code, name, or partial) to airport code.

    Returns tuple (airport_code, matches_list) where matches_list is None if unique match,
    or a list of (code, name) tuples if ambiguous/not found.
    """
    input_upper = input_str.upper()

    # Exact code match
    if input_upper in AIRPORTS:
        return input_upper, None

    # Exact name match (case-insensitive)
    for code, name in AIRPORTS.items():
        if name.upper() == input_upper:
            return code, None

    # Partial match - check if input matches start of code or name
    matches = []
    for code, name in AIRPORTS.items():
        if code.startswith(input_upper) or name.upper().startswith(input_upper):
            matches.append((code, name))

    # If exactly one match, return it
    if len(matches) == 1:
        return matches[0][0], None

    # If multiple matches or no matches, return None code with matches list
    return None, matches


def format_airports_list() -> str:
    """Format airports list for display with both name and code."""
    items = []
    for code, name in sorted(AIRPORTS.items()):
        items.append(f"{name} ({code})")
    return ", ".join(items)


def get_nepal_date() -> date:
    """Get today's date in Nepal timezone.

    Returns:
        Current date in Nepal timezone (Asia/Kathmandu)
    """
    return datetime.now(NEPAL_TZ).date()


def parse_xml(xml_content: str, flight_date: Optional[date] = None) -> List[Dict]:
    """Parse XML content and extract flight information.

    Args:
        xml_content: XML string from API
        flight_date: Date for the flights (defaults to today)

    Returns:
        List of flight dictionaries
    """
    if flight_date is None:
        flight_date = get_nepal_date()

    root = ET.fromstring(xml_content)
    flights = []

    for flight_elem in root.findall("Flight"):
        flight_data = {
            "flight_no": flight_elem.findtext("FlightNo", ""),
            "departure": flight_elem.findtext("Departure", ""),
            "arrival": flight_elem.findtext("Arrival", ""),
            "flight_time": flight_elem.findtext("FlightTime", ""),
            "revised_time": flight_elem.findtext("RevisedTime", ""),
            "flight_status": flight_elem.findtext("FlightStatus", ""),
            "flight_remarks": flight_elem.findtext("FlightRemarks", ""),
            "flight_date": flight_date,
        }
        flights.append(flight_data)

    return flights


def store_flights(conn, flights: List[Dict]) -> Tuple[int, int]:
    """Store flights in the database as a log (always insert, never update).
    Skips inserting if exact same data already exists for the same date.

    Returns:
        Tuple of (inserted_count, skipped_count)
    """
    cursor = conn.cursor()

    inserted_count = 0
    skipped_count = 0

    for flight in flights:
        flight_date = flight.get("flight_date", get_nepal_date())
        # Convert date to string to avoid deprecation warning in Python 3.12+
        flight_date_str = (
            flight_date.isoformat() if isinstance(flight_date, date) else str(flight_date)
        )

        # Check if exact same flight data already exists for this date
        cursor.execute(
            """
            SELECT id FROM flights
            WHERE flight_no = ? AND departure = ? AND arrival = ? AND flight_date = ?
            AND flight_time = ? AND revised_time = ? AND flight_status = ? AND flight_remarks = ?
            LIMIT 1
        """,
            (
                flight["flight_no"],
                flight["departure"],
                flight["arrival"],
                flight_date_str,
                flight.get("flight_time", "") or "",
                flight.get("revised_time", "") or "",
                flight.get("flight_status", "") or "",
                flight.get("flight_remarks", "") or "",
            ),
        )

        existing = cursor.fetchone()

        if existing:
            # Exact duplicate, skip
            skipped_count += 1
        else:
            # Insert new record (even if flight exists with different data - log it)
            cursor.execute(
                """
                INSERT INTO flights
                (flight_no, departure, arrival, flight_time, revised_time,
                 flight_status, flight_remarks, flight_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    flight["flight_no"],
                    flight["departure"],
                    flight["arrival"],
                    flight["flight_time"],
                    flight["revised_time"],
                    flight["flight_status"],
                    flight["flight_remarks"],
                    flight_date_str,
                ),
            )
            inserted_count += 1

    return flights


async def fetch_all_combinations_async(
    conn,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[int, int, int]:
    """Fetch flight status for all airport combinations asynchronously.

    Args:
        conn: Database connection
        max_concurrent: Maximum number of concurrent requests
        progress_callback: Optional callback function for progress messages

    Returns:
        Tuple of (successful_routes, failed_routes, total_flights)
    """
    airport_codes = get_airport_codes()

    if not airport_codes:
        if progress_callback:
            progress_callback("[yellow]No airports found.[/yellow]")
        return (0, 0, 0)

    # Generate all route combinations
    routes = [
        (dep, arr)
        for dep in airport_codes
        for arr in airport_codes
        if dep != arr
    ]

    total_combinations = len(routes)
    if progress_callback:
        progress_callback(
            f"[cyan]Fetching flight status for {total_combinations} route combinations...[/cyan]\n"
        )

    total_flights = 0
    successful_routes = 0
    failed_routes = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_and_store_route(departure: str, arrival: str) -> Tuple[int, int, bool]:
        """Fetch a single route and store results."""
        async with semaphore:
            try:
                async with httpx.AsyncClient() as client:
                    xml_content = await fetch_flight_status_async(client, departure, arrival)

                    # Check if response is valid XML (not empty or error)
                    if not xml_content or xml_content.strip() == "":
                        return (0, 0, False)

                    # Check for JSON error responses
                    if xml_content.strip().startswith("{"):
                        return (0, 0, False)

                    # Parse XML
                    try:
                        flights = parse_xml(xml_content)
                    except Exception:
                        return (0, 0, False)

                    if flights:
                        # Store flights (database operations are synchronous)
                        inserted, skipped = store_flights(conn, flights)
                        return (inserted, skipped, True)
                    else:
                        return (0, 0, False)

            except Exception:
                return (0, 0, False)

    # Create tasks for all routes
    tasks = [
        fetch_and_store_route(dep, arr) for dep, arr in routes
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, result in enumerate(results):
        departure, arrival = routes[i]
        route_num = i + 1

        if isinstance(result, Exception):
            failed_routes += 1
            if progress_callback:
                msg = (
                    f"[dim][{route_num}/{total_combinations}][/dim] "
                    f"{departure} → {arrival}: [red]✗ Error[/red]"
                )
                progress_callback(msg)
        else:
            inserted, skipped, success = result
            if success:
                successful_routes += 1
                total_flights += inserted
                if progress_callback:
                    status_msg = (
                        f"[dim][{route_num}/{total_combinations}][/dim] "
                        f"{departure} → {arrival}: "
                    )
                    if inserted > 0:
                        status_parts = [f"{inserted} new"]
                        if skipped > 0:
                            status_parts.append(f"{skipped} skipped")
                        progress_callback(f"{status_msg}[green]✓ {', '.join(status_parts)}[/green]")
                    else:
                        progress_callback(
                            f"{status_msg}[dim]No new data ({skipped} skipped)[/dim]"
                        )
            else:
                failed_routes += 1
                if progress_callback:
                    status_msg = (
                        f"[dim][{route_num}/{total_combinations}][/dim] "
                        f"{departure} → {arrival}: "
                    )
                    progress_callback(f"{status_msg}[dim]No flights[/dim]")

    return (successful_routes, failed_routes, total_flights)


def get_flights_from_db(
    conn,
    departure_code: Optional[str] = None,
    arrival_code: Optional[str] = None,
    limit: Optional[int] = None,
    flight_date: Optional[date] = None,
) -> List[Dict]:
    """Get flights from the database.

    Args:
        departure_code: Airport code (e.g., 'KTM') - will match against airport name
        arrival_code: Airport code (e.g., 'BHR') - will match against airport name
        limit: Maximum number of flights to return
        flight_date: Filter by specific date (if None, returns all dates)

    Returns:
        List of flight dictionaries
    """
    cursor = conn.cursor()

    conditions = []
    params = []

    if departure_code:
        # Match by airport name (database stores names like "KATHMANDU")
        # Remove parenthetical info and match base name
        departure_name = AIRPORTS.get(departure_code, "").upper()
        # Remove anything in parentheses for matching
        departure_name_base = departure_name.split("(")[0].strip()
        conditions.append("UPPER(departure) LIKE ?")
        params.append(f"{departure_name_base}%")

    if arrival_code:
        # Match by airport name (database stores names like "BHARATPUR")
        arrival_name = AIRPORTS.get(arrival_code, "").upper()
        # Remove anything in parentheses for matching
        arrival_name_base = arrival_name.split("(")[0].strip()
        conditions.append("UPPER(arrival) LIKE ?")
        params.append(f"{arrival_name_base}%")

    if flight_date:
        conditions.append("flight_date = ?")
        params.append(
            flight_date.isoformat() if isinstance(flight_date, date) else str(flight_date)
        )

    # Build WHERE clause with table prefix
    where_clause_base = ""
    where_clause_filtered = ""
    if conditions:
        # For the CTE, use base table
        where_clause_base = " WHERE " + " AND ".join(conditions)
        # For the final SELECT, use table alias
        filtered_conditions = [
            c.replace("UPPER(departure)", "UPPER(f.departure)").replace(
                "UPPER(arrival)", "UPPER(f.arrival)"
            )
            for c in conditions
        ]
        where_clause_filtered = " WHERE " + " AND ".join(filtered_conditions)

    # Deduplicate: get only the latest record for each
    # flight_no + departure + arrival + flight_time combination
    # Use a CTE to find the latest fetched_at for each unique flight route and time
    deduplicated_query = f"""
        WITH latest_flights AS (
            SELECT flight_no, departure, arrival, flight_time, MAX(fetched_at) as max_fetched_at
            FROM flights
            {where_clause_base}
            GROUP BY flight_no, departure, arrival, flight_time
        )
        SELECT f.flight_no, f.departure, f.arrival, f.flight_time, f.revised_time,
               f.flight_status, f.flight_remarks, f.flight_date, f.fetched_at
        FROM flights f
        INNER JOIN latest_flights l ON f.flight_no = l.flight_no
            AND f.departure = l.departure
            AND f.arrival = l.arrival
            AND f.flight_time = l.flight_time
            AND f.fetched_at = l.max_fetched_at
        {where_clause_filtered}
        ORDER BY f.flight_date DESC, f.departure, f.arrival, f.flight_time
    """

    if limit:
        deduplicated_query += f" LIMIT {limit}"

    # Duplicate params for both WHERE clauses
    params_deduplicated = params + params if conditions else []

    cursor.execute(deduplicated_query, params_deduplicated)
    rows = cursor.fetchall()

    flights = []
    for row in rows:
        flights.append(
            {
                "flight_no": row[0],
                "departure": row[1],
                "arrival": row[2],
                "flight_time": row[3],
                "revised_time": row[4],
                "flight_status": row[5],
                "flight_remarks": row[6],
                "flight_date": row[7],
                "fetched_at": row[8],
            }
        )

    return flights


def get_unique_flight_routes(conn, flight_date: Optional[date] = None) -> List[Dict[str, str]]:
    """Get unique flight routes (departure + arrival + flight_no) from database.

    Args:
        flight_date: Filter by specific date (if None, returns all dates)

    Returns:
        List of dictionaries with 'departure', 'arrival', 'flight_no' keys
    """
    cursor = conn.cursor()

    conditions = []
    params = []

    if flight_date:
        conditions.append("flight_date = ?")
        params.append(
            flight_date.isoformat() if isinstance(flight_date, date) else str(flight_date)
        )

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT DISTINCT flight_no, departure, arrival
        FROM flights
        {where_clause}
        ORDER BY departure, arrival, flight_no
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    routes = []
    for row in rows:
        routes.append(
            {
                "flight_no": row[0],
                "departure": row[1],
                "arrival": row[2],
            }
        )

    return routes


def get_flights_for_route(
    conn,
    departure: str,
    arrival: str,
    flight_no: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict]:
    """Get historical flights for a specific route and flight number.

    Args:
        departure: Departure airport name (as stored in DB)
        arrival: Arrival airport name (as stored in DB)
        flight_no: Flight number
        start_date: Start date for date range (inclusive)
        end_date: End date for date range (inclusive)

    Returns:
        List of flight dictionaries ordered by date
    """
    cursor = conn.cursor()

    conditions = ["flight_no = ?", "departure = ?", "arrival = ?"]
    params = [flight_no, departure, arrival]

    if start_date:
        conditions.append("flight_date >= ?")
        params.append(start_date.isoformat() if isinstance(start_date, date) else str(start_date))

    if end_date:
        conditions.append("flight_date <= ?")
        params.append(end_date.isoformat() if isinstance(end_date, date) else str(end_date))

    where_clause = " WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT flight_no, departure, arrival, flight_time, revised_time,
               flight_status, flight_remarks, flight_date, fetched_at
        FROM flights
        {where_clause}
        ORDER BY flight_date DESC, flight_time
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    flights = []
    for row in rows:
        flights.append(
            {
                "flight_no": row[0],
                "departure": row[1],
                "arrival": row[2],
                "flight_time": row[3],
                "revised_time": row[4],
                "flight_status": row[5],
                "flight_remarks": row[6],
                "flight_date": row[7],
                "fetched_at": row[8],
            }
        )

    return flights
