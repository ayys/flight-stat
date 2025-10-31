"""Core library functions for flight status management.

This module provides framework-agnostic functions for database operations,
API interactions, and data processing. No Rich console dependencies.
"""

import asyncio
import os
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Tuple

import asyncpg
import httpx
import requests
from tqdm.asyncio import tqdm as atqdm
from zoneinfo import ZoneInfo

# Nepal timezone (UTC+5:45)
NEPAL_TZ = ZoneInfo("Asia/Kathmandu")

# Database connection string (NeonDB PostgreSQL)
DB_CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    None,
)

# Global connection pool
_db_pool: Optional[asyncpg.Pool] = None

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


async def init_database() -> asyncpg.Pool:
    """Initialize PostgreSQL database connection pool and create tables if they don't exist.

    Returns:
        Connection pool instance
    """
    global _db_pool

    if _db_pool is not None:
        return _db_pool

    if not DB_CONNECTION_STRING:
        raise ValueError("DATABASE_URL environment variable is not set")

    # Create connection pool with reasonable defaults
    _db_pool = await asyncpg.create_pool(
        DB_CONNECTION_STRING,
        min_size=2,
        max_size=20,
        command_timeout=60,
    )

    # Use a connection from the pool to set up tables
    async with _db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS flights (
                id SERIAL PRIMARY KEY,
                flight_no VARCHAR NOT NULL,
                departure VARCHAR NOT NULL,
                arrival VARCHAR NOT NULL,
                flight_time VARCHAR NOT NULL,
                revised_time VARCHAR NOT NULL,
                flight_status VARCHAR,
                flight_remarks VARCHAR,
                flight_date DATE NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Check if flight_date column exists (for existing databases)
        columns = await conn.fetch("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'flights' AND column_name = 'flight_date'
        """)
        if not columns:
            await conn.execute("ALTER TABLE flights ADD COLUMN flight_date DATE")
            # Set default date for existing records
            await conn.execute("UPDATE flights SET flight_date = CAST(fetched_at AS DATE) WHERE flight_date IS NULL")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS airports (
                id SERIAL PRIMARY KEY,
                code VARCHAR UNIQUE NOT NULL,
                name VARCHAR NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table to track routes that have no flights (to avoid unnecessary API calls)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS no_flight_routes (
                id SERIAL PRIMARY KEY,
                departure VARCHAR NOT NULL,
                arrival VARCHAR NOT NULL,
                last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(departure, arrival)
            )
        """)

    return _db_pool


async def close_database_pool() -> None:
    """Close the database connection pool."""
    global _db_pool
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None


def _get_connection_or_pool(conn_or_pool):
    """Helper to handle both connection and pool.

    Returns tuple (is_pool, connection_or_pool)
    """
    if isinstance(conn_or_pool, asyncpg.Pool):
        return True, conn_or_pool
    return False, conn_or_pool


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


async def get_all_flagged_routes(conn_or_pool) -> set:
    """Get all routes flagged as having no flights.

    Args:
        conn_or_pool: Database connection or pool

    Returns:
        Set of (departure, arrival) tuples
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)

    if is_pool:
        async with db.acquire() as conn:
            rows = await conn.fetch("""
                SELECT departure, arrival FROM no_flight_routes
            """)
    else:
        rows = await db.fetch("""
            SELECT departure, arrival FROM no_flight_routes
        """)

    return {(row["departure"], row["arrival"]) for row in rows}


async def is_route_flagged_no_flights(conn_or_pool, departure: str, arrival: str) -> bool:
    """Check if a route is flagged as having no flights.

    Args:
        conn_or_pool: Database connection or pool
        departure: Departure airport code
        arrival: Arrival airport code

    Returns:
        True if route is flagged as having no flights
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)

    if is_pool:
        async with db.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT id FROM no_flight_routes
                WHERE departure = $1 AND arrival = $2
                LIMIT 1
            """,
                departure,
                arrival,
            )
            return result is not None
    else:
        result = await db.fetchval(
            """
            SELECT id FROM no_flight_routes
            WHERE departure = $1 AND arrival = $2
            LIMIT 1
        """,
            departure,
            arrival,
        )
        return result is not None


async def flag_route_no_flights(conn_or_pool, departure: str, arrival: str) -> None:
    """Flag a route as having no flights.

    Args:
        conn_or_pool: Database connection or pool
        departure: Departure airport code
        arrival: Arrival airport code
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)

    if is_pool:
        async with db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO no_flight_routes (departure, arrival, last_checked)
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT (departure, arrival) DO UPDATE SET last_checked = CURRENT_TIMESTAMP
            """,
                departure,
                arrival,
            )
    else:
        await db.execute(
            """
            INSERT INTO no_flight_routes (departure, arrival, last_checked)
            VALUES ($1, $2, CURRENT_TIMESTAMP)
            ON CONFLICT (departure, arrival) DO UPDATE SET last_checked = CURRENT_TIMESTAMP
        """,
            departure,
            arrival,
        )


async def analyze_existing_routes(conn_or_pool, progress_callback: Optional[Callable[[str], None]] = None) -> int:
    """Analyze existing flight data to count routes that haven't been fetched yet.

    This checks which routes exist in the database vs all possible routes.
    Note: Routes are only flagged as "no flights" after actually checking the API,
    not based on database presence alone.

    Args:
        conn_or_pool: Database connection or pool
        progress_callback: Optional callback function for progress messages

    Returns:
        Number of routes not yet fetched (for informational purposes only)
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)

    if progress_callback:
        progress_callback("  [dim]Querying database for existing routes...[/dim]")

    # Get all unique routes from flights table
    if is_pool:
        async with db.acquire() as conn:
            routes_with_flights = await conn.fetch("""
                SELECT DISTINCT departure, arrival
                FROM flights
            """)
    else:
        routes_with_flights = await db.fetch("""
            SELECT DISTINCT departure, arrival
            FROM flights
        """)

    # Exit early if no records exist
    if not routes_with_flights:
        if progress_callback:
            progress_callback("  [dim]No existing routes found in database[/dim]")
        return 0

    if progress_callback:
        progress_callback(f"  [dim]Found {len(routes_with_flights)} unique routes in database[/dim]")

    routes_with_flights_set = {(row["departure"], row["arrival"]) for row in routes_with_flights}

    # Get all airport combinations
    airport_codes = get_airport_codes()
    all_routes = [(dep, arr) for dep in airport_codes for arr in airport_codes if dep != arr]

    if progress_callback:
        progress_callback(f"  [dim]Checking {len(all_routes)} possible route combinations...[/dim]")

    # Count routes that don't have any flights in database yet
    # (Note: These aren't flagged - flagging only happens after API checks)
    # Convert departure/arrival to airport codes for matching
    unfetched_count = 0
    for dep_code, arr_code in all_routes:
        # Check if this route has any flights in the database
        # We need to match airport names, not codes
        dep_name = AIRPORTS.get(dep_code, "").upper().split("(")[0].strip()
        arr_name = AIRPORTS.get(arr_code, "").upper().split("(")[0].strip()

        has_flights = any(dep_name in stored_dep.upper() and arr_name in stored_arr.upper() for stored_dep, stored_arr in routes_with_flights_set)

        if not has_flights:
            unfetched_count += 1

    return unfetched_count


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


async def store_flights(conn_or_pool, flights: List[Dict], progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[int, int]:
    """Store flights in the database as a log (always insert, never update).
    Skips inserting if exact same data already exists for the same date.

    Args:
        conn_or_pool: Database connection or pool
        flights: List of flight dictionaries to store
        progress_callback: Optional callback function for progress messages

    Returns:
        Tuple of (inserted_count, skipped_count)
    """
    if not flights:
        return (0, 0)

    if progress_callback:
        progress_callback(f"  [dim]Processing {len(flights)} flight(s) for storage...[/dim]")

    is_pool, db = _get_connection_or_pool(conn_or_pool)

    # Normalize flight dates and prepare flight data
    normalized_flights = []
    for flight in flights:
        flight_date = flight.get("flight_date", get_nepal_date())
        # Ensure flight_date is a date object
        if isinstance(flight_date, str):
            from datetime import datetime

            flight_date = datetime.strptime(flight_date, "%Y-%m-%d").date()
        elif not isinstance(flight_date, date):
            flight_date = get_nepal_date()

        normalized_flights.append(
            {
                "flight_no": flight["flight_no"],
                "departure": flight["departure"],
                "arrival": flight["arrival"],
                "flight_time": flight.get("flight_time", "") or "",
                "revised_time": flight.get("revised_time", "") or "",
                "flight_status": flight.get("flight_status", "") or "",
                "flight_remarks": flight.get("flight_remarks", "") or "",
                "flight_date": flight_date,
            }
        )

    # Bulk check for existing flights using a single query with VALUES and JOIN
    async def _check_existing(conn):
        if not normalized_flights:
            return set()

        # Build VALUES clause for all flights
        values_clauses = []
        params = []
        param_num = 1

        for flight in normalized_flights:
            values_clauses.append(
                f"(${param_num}, ${param_num + 1}, ${param_num + 2}, ${param_num + 3}::DATE, "
                f"${param_num + 4}, ${param_num + 5}, ${param_num + 6}, ${param_num + 7})"
            )
            params.extend(
                [
                    flight["flight_no"],
                    flight["departure"],
                    flight["arrival"],
                    flight["flight_date"],
                    flight["flight_time"],
                    flight["revised_time"],
                    flight["flight_status"],
                    flight["flight_remarks"],
                ]
            )
            param_num += 8

        # Use VALUES with JOIN to check existence efficiently
        query = f"""
            SELECT f.flight_no, f.departure, f.arrival, f.flight_date, f.flight_time,
                   f.revised_time, f.flight_status, f.flight_remarks
            FROM flights f
            INNER JOIN (VALUES {", ".join(values_clauses)}) AS v
                (flight_no, departure, arrival, flight_date, flight_time,
                 revised_time, flight_status, flight_remarks)
            ON f.flight_no = v.flight_no
                AND f.departure = v.departure
                AND f.arrival = v.arrival
                AND f.flight_date = v.flight_date
                AND f.flight_time = v.flight_time
                AND f.revised_time = v.revised_time
                AND f.flight_status = v.flight_status
                AND f.flight_remarks = v.flight_remarks
        """

        existing_rows = await conn.fetch(query, *params)

        # Create a set of tuples for fast lookup
        existing_set = {
            (
                row["flight_no"],
                row["departure"],
                row["arrival"],
                row["flight_date"],
                row["flight_time"],
                row["revised_time"],
                row["flight_status"],
                row["flight_remarks"],
            )
            for row in existing_rows
        }

        return existing_set

    # Separate flights into new and existing
    if is_pool:
        async with db.acquire() as conn:
            existing_set = await _check_existing(conn)
    else:
        existing_set = await _check_existing(db)

    new_flights = []
    skipped_count = 0

    for flight in normalized_flights:
        flight_key = (
            flight["flight_no"],
            flight["departure"],
            flight["arrival"],
            flight["flight_date"],
            flight["flight_time"],
            flight["revised_time"],
            flight["flight_status"],
            flight["flight_remarks"],
        )

        if flight_key in existing_set:
            skipped_count += 1
        else:
            new_flights.append(flight)

    # Bulk insert new flights
    inserted_count = 0
    if new_flights:
        if is_pool:
            async with db.acquire() as conn:
                # Use executemany for bulk insert
                await conn.executemany(
                    """
                    INSERT INTO flights
                    (flight_no, departure, arrival, flight_time, revised_time,
                     flight_status, flight_remarks, flight_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    [
                        (
                            flight["flight_no"],
                            flight["departure"],
                            flight["arrival"],
                            flight["flight_time"],
                            flight["revised_time"],
                            flight["flight_status"],
                            flight["flight_remarks"],
                            flight["flight_date"],
                        )
                        for flight in new_flights
                    ],
                )
                inserted_count = len(new_flights)
        else:
            await db.executemany(
                """
                INSERT INTO flights
                (flight_no, departure, arrival, flight_time, revised_time,
                 flight_status, flight_remarks, flight_date)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                [
                    (
                        flight["flight_no"],
                        flight["departure"],
                        flight["arrival"],
                        flight["flight_time"],
                        flight["revised_time"],
                        flight["flight_status"],
                        flight["flight_remarks"],
                        flight["flight_date"],
                    )
                    for flight in new_flights
                ],
            )
            inserted_count = len(new_flights)

    if progress_callback:
        progress_callback(f"  [dim]Storage complete: {inserted_count} inserted, {skipped_count} skipped[/dim]")

    return inserted_count, skipped_count


async def fetch_all_combinations_async(
    conn_or_pool,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_tqdm: bool = True,
) -> Tuple[int, int, int]:
    """Fetch flight status for all airport combinations asynchronously.

    Args:
        conn_or_pool: Database connection or pool
        max_concurrent: Maximum number of concurrent requests
        progress_callback: Optional callback function for progress messages (used if use_tqdm=False)
        use_tqdm: Whether to use tqdm progress bar (default: True)

    Returns:
        Tuple of (successful_routes, failed_routes, total_flights)
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)

    if progress_callback:
        progress_callback("  [dim]Getting airport codes...[/dim]")
    airport_codes = get_airport_codes()

    if not airport_codes:
        if progress_callback:
            progress_callback("[yellow]No airports found.[/yellow]")
        return (0, 0, 0)

    if progress_callback:
        progress_callback(f"  [dim]Found {len(airport_codes)} airports[/dim]")

    # Generate all route combinations
    if progress_callback:
        progress_callback("  [dim]Generating route combinations...[/dim]")
    routes = [(dep, arr) for dep in airport_codes for arr in airport_codes if dep != arr]

    if progress_callback:
        progress_callback(f"  [dim]Total possible routes: {len(routes)}[/dim]")

    # Filter out routes that are flagged as having no flights
    if progress_callback:
        progress_callback("  [dim]Filtering out routes flagged as having no flights...[/dim]")
    flagged_routes = await get_all_flagged_routes(conn_or_pool)
    routes_to_fetch = [(dep, arr) for dep, arr in routes if (dep, arr) not in flagged_routes]
    skipped_count = len(routes) - len(routes_to_fetch)

    total_combinations = len(routes_to_fetch)
    if progress_callback and not use_tqdm:
        if skipped_count > 0:
            msg = (
                f"  [cyan]Fetching flight status for {total_combinations} route combinations "
                f"({skipped_count} skipped routes with no flights)...[/cyan]\n"
            )
            progress_callback(msg)
        else:
            msg = f"  [cyan]Fetching flight status for {total_combinations} route combinations...[/cyan]\n"
            progress_callback(msg)

    total_flights = 0
    successful_routes = 0
    failed_routes = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_and_store_route(
        departure: str, arrival: str, pbar: Optional[atqdm] = None, route_index: int = 0, total_routes: int = 0
    ) -> Tuple[int, int, bool]:
        """Fetch a single route and store results."""
        async with semaphore:
            try:
                async with httpx.AsyncClient() as client:
                    xml_content = await fetch_flight_status_async(client, departure, arrival)

                    # Small delay to be respectful to the API
                    await asyncio.sleep(0.05)

                    # Check if response is valid XML (not empty or error)
                    if not xml_content or xml_content.strip() == "":
                        # Flag route as having no flights
                        await flag_route_no_flights(conn_or_pool, departure, arrival)
                        if pbar:
                            pbar.set_postfix_str(f"{departure} → {arrival}: No data")
                        elif progress_callback and not use_tqdm:
                            progress_callback(f"  [dim][{route_index + 1}/{total_routes}][/dim] {departure} → {arrival}: [dim]No data[/dim]")
                        return (0, 0, False)

                    # Check for JSON error responses
                    if xml_content.strip().startswith("{"):
                        if pbar:
                            pbar.set_postfix_str(f"{departure} → {arrival}: API error")
                        elif progress_callback and not use_tqdm:
                            progress_callback(f"  [dim][{route_index + 1}/{total_routes}][/dim] {departure} → {arrival}: [red]API error[/red]")
                        return (0, 0, False)

                    # Parse XML
                    try:
                        flights = parse_xml(xml_content)
                    except Exception as e:
                        error_msg = f"{departure} → {arrival}: Invalid XML: {str(e)}"
                        if pbar:
                            pbar.set_postfix_str(error_msg)
                        if progress_callback:
                            progress_callback(f"  [red]✗ {error_msg}[/red]")
                        return (0, 0, False)

                    if flights:
                        # Store flights (database operations are async)
                        # Each concurrent task uses the pool, which will acquire its own connection
                        try:
                            inserted, skipped = await store_flights(conn_or_pool, flights, progress_callback=None)
                            if pbar:
                                status = f"{inserted} new" if inserted > 0 else f"{skipped} skipped"
                                pbar.set_postfix_str(f"{departure} → {arrival}: {status}")
                            elif progress_callback and not use_tqdm:
                                status_msg = f"  [dim][{route_index + 1}/{total_routes}][/dim] {departure} → {arrival}: "
                                if inserted > 0:
                                    status_parts = [f"{inserted} new"]
                                    if skipped > 0:
                                        status_parts.append(f"{skipped} skipped")
                                    progress_callback(f"{status_msg}[green]✓ {', '.join(status_parts)}[/green]")
                                else:
                                    progress_callback(f"{status_msg}[dim]No new data ({skipped} skipped)[/dim]")
                            return (inserted, skipped, True)
                        except Exception as store_error:
                            # Re-raise to be caught by outer handler
                            raise Exception(f"Error storing flights for {departure} → {arrival}: {str(store_error)}") from store_error
                    else:
                        # Flag route as having no flights
                        await flag_route_no_flights(conn_or_pool, departure, arrival)
                        if pbar:
                            pbar.set_postfix_str(f"{departure} → {arrival}: No flights")
                        elif progress_callback and not use_tqdm:
                            progress_callback(f"  [dim][{route_index + 1}/{total_routes}][/dim] {departure} → {arrival}: [dim]No flights[/dim]")
                        return (0, 0, False)

            except Exception as e:
                error_msg = f"{departure} → {arrival}: Error: {str(e)}"
                if pbar:
                    pbar.set_postfix_str(error_msg)
                if progress_callback:
                    # Always log errors, even with tqdm
                    progress_callback(f"  [red]✗ {error_msg}[/red]")
                # Log full exception for debugging
                import traceback

                if progress_callback:
                    progress_callback(f"  [dim]{traceback.format_exc()}[/dim]")
                return (0, 0, False)

    # Create progress bar
    pbar = atqdm(total=total_combinations, desc="Fetching routes", unit="route") if use_tqdm else None

    # Create tasks for all routes
    tasks = [fetch_and_store_route(dep, arr, pbar, idx, total_combinations) for idx, (dep, arr) in enumerate(routes_to_fetch)]

    # Execute all tasks concurrently with progress tracking
    results = []
    if use_tqdm:
        # Wrap tasks to update progress bar
        async def wrap_task(task):
            result = await task
            if pbar:
                pbar.update(1)
            return result

        wrapped_tasks = [wrap_task(task) for task in tasks]
        results = await asyncio.gather(*wrapped_tasks, return_exceptions=True)
    else:
        # Use regular gather and manual progress updates
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Close progress bar
    if pbar:
        pbar.close()

    # Process results (logging is now done in fetch_and_store_route)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed_routes += 1
            if progress_callback:
                progress_callback(f"  [red]Route {i + 1} raised exception: {str(result)}[/red]")
        else:
            inserted, skipped, success = result
            if success:
                successful_routes += 1
                total_flights += inserted
            else:
                failed_routes += 1
                if progress_callback and inserted == 0 and skipped == 0:
                    # Log why route failed if no flights were processed
                    departure, arrival = routes_to_fetch[i]
                    progress_callback(f"  [yellow]Route {i + 1} ({departure} → {arrival}): No flights found or error[/yellow]")

    if progress_callback:
        progress_callback("")

    return (successful_routes, failed_routes, total_flights)


async def get_flights_from_db(
    conn_or_pool,
    departure_code: Optional[str] = None,
    arrival_code: Optional[str] = None,
    limit: Optional[int] = None,
    flight_date: Optional[date] = None,
) -> List[Dict]:
    """Get flights from the database.

    Args:
        conn_or_pool: Database connection or pool
        departure_code: Airport code (e.g., 'KTM') - will match against airport name
        arrival_code: Airport code (e.g., 'BHR') - will match against airport name
        limit: Maximum number of flights to return
        flight_date: Filter by specific date (if None, returns all dates)

    Returns:
        List of flight dictionaries
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)
    conditions_base = []
    conditions_filtered = []
    params_base = []
    param_num = 1

    if departure_code:
        departure_name = AIRPORTS.get(departure_code, "").upper()
        departure_name_base = departure_name.split("(")[0].strip()
        conditions_base.append(f"UPPER(departure) LIKE ${param_num}")
        params_base.append(f"{departure_name_base}%")
        param_num += 1

    if arrival_code:
        arrival_name = AIRPORTS.get(arrival_code, "").upper()
        arrival_name_base = arrival_name.split("(")[0].strip()
        conditions_base.append(f"UPPER(arrival) LIKE ${param_num}")
        params_base.append(f"{arrival_name_base}%")
        param_num += 1

    if flight_date:
        # Ensure flight_date is a date object
        if isinstance(flight_date, str):
            from datetime import datetime

            flight_date = datetime.strptime(flight_date, "%Y-%m-%d").date()
        elif not isinstance(flight_date, date):
            flight_date = get_nepal_date()
        conditions_base.append(f"flight_date = ${param_num}")
        params_base.append(flight_date)  # Pass date object directly
        param_num += 1

    # Build filtered conditions with table alias and renumbered parameters
    filtered_param_num = param_num
    for i, c in enumerate(conditions_base):
        filtered_cond = (
            c.replace("UPPER(departure)", "UPPER(f.departure)")
            .replace("UPPER(arrival)", "UPPER(f.arrival)")
            .replace(f"${i + 1}", f"${filtered_param_num}")
        )
        conditions_filtered.append(filtered_cond)
        filtered_param_num += 1

    where_clause_base = ""
    where_clause_filtered = ""
    if conditions_base:
        where_clause_base = " WHERE " + " AND ".join(conditions_base)
        where_clause_filtered = " WHERE " + " AND ".join(conditions_filtered)

    # Duplicate params for both WHERE clauses
    params_deduplicated = params_base + params_base if conditions_base else []

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

    if is_pool:
        async with db.acquire() as conn:
            rows = await conn.fetch(deduplicated_query, *params_deduplicated)
    else:
        rows = await db.fetch(deduplicated_query, *params_deduplicated)

    flights = []
    for row in rows:
        flights.append(
            {
                "flight_no": row["flight_no"],
                "departure": row["departure"],
                "arrival": row["arrival"],
                "flight_time": row["flight_time"],
                "revised_time": row["revised_time"],
                "flight_status": row["flight_status"],
                "flight_remarks": row["flight_remarks"],
                "flight_date": row["flight_date"],
                "fetched_at": row["fetched_at"],
            }
        )

    return flights


async def get_unique_flight_routes(conn_or_pool, flight_date: Optional[date] = None) -> List[Dict[str, str]]:
    """Get unique flight routes (departure + arrival + flight_no) from database.

    Args:
        conn_or_pool: Database connection or pool
        flight_date: Filter by specific date (if None, returns all dates)

    Returns:
        List of dictionaries with 'departure', 'arrival', 'flight_no' keys
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)
    conditions = []
    params = []
    param_num = 1

    if flight_date:
        # Ensure flight_date is a date object
        if isinstance(flight_date, str):
            from datetime import datetime

            flight_date = datetime.strptime(flight_date, "%Y-%m-%d").date()
        elif not isinstance(flight_date, date):
            flight_date = get_nepal_date()
        conditions.append(f"flight_date = ${param_num}")
        params.append(flight_date)  # Pass date object directly
        param_num += 1

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT DISTINCT flight_no, departure, arrival
        FROM flights
        {where_clause}
        ORDER BY departure, arrival, flight_no
    """

    if is_pool:
        async with db.acquire() as conn:
            rows = await conn.fetch(query, *params)
    else:
        rows = await db.fetch(query, *params)

    routes = []
    for row in rows:
        routes.append(
            {
                "flight_no": row["flight_no"],
                "departure": row["departure"],
                "arrival": row["arrival"],
            }
        )

    return routes


async def get_flights_for_route(
    conn_or_pool,
    departure: str,
    arrival: str,
    flight_no: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> List[Dict]:
    """Get historical flights for a specific route and flight number.

    Args:
        conn_or_pool: Database connection or pool
        departure: Departure airport name (as stored in DB)
        arrival: Arrival airport name (as stored in DB)
        flight_no: Flight number
        start_date: Start date for date range (inclusive)
        end_date: End date for date range (inclusive)

    Returns:
        List of flight dictionaries ordered by date
    """
    is_pool, db = _get_connection_or_pool(conn_or_pool)
    conditions = ["flight_no = $1", "departure = $2", "arrival = $3"]
    params = [flight_no, departure, arrival]
    param_num = 4

    if start_date:
        # Ensure start_date is a date object
        if isinstance(start_date, str):
            from datetime import datetime

            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        elif not isinstance(start_date, date):
            start_date = get_nepal_date()
        conditions.append(f"flight_date >= ${param_num}")
        params.append(start_date)  # Pass date object directly
        param_num += 1

    if end_date:
        # Ensure end_date is a date object
        if isinstance(end_date, str):
            from datetime import datetime

            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        elif not isinstance(end_date, date):
            end_date = get_nepal_date()
        conditions.append(f"flight_date <= ${param_num}")
        params.append(end_date)  # Pass date object directly
        param_num += 1

    where_clause = " WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT flight_no, departure, arrival, flight_time, revised_time,
               flight_status, flight_remarks, flight_date, fetched_at
        FROM flights
        {where_clause}
        ORDER BY flight_date DESC, flight_time
    """

    if is_pool:
        async with db.acquire() as conn:
            rows = await conn.fetch(query, *params)
    else:
        rows = await db.fetch(query, *params)

    flights = []
    for row in rows:
        flights.append(
            {
                "flight_no": row["flight_no"],
                "departure": row["departure"],
                "arrival": row["arrival"],
                "flight_time": row["flight_time"],
                "revised_time": row["revised_time"],
                "flight_status": row["flight_status"],
                "flight_remarks": row["flight_remarks"],
                "flight_date": row["flight_date"],
                "fetched_at": row["fetched_at"],
            }
        )

    return flights
