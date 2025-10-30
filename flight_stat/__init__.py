"""
Flight Status library for fetching and managing Buddha Air flight data.
"""

from flight_stat.lib import (
    AIRPORTS,
    DB_PATH,
    fetch_flight_status,
    format_airports_list,
    get_airport_codes,
    get_flights_for_route,
    get_flights_from_db,
    get_unique_flight_routes,
    init_database,
    match_airport,
    parse_xml,
    store_flights,
)

__all__ = [
    "AIRPORTS",
    "DB_PATH",
    "init_database",
    "fetch_flight_status",
    "parse_xml",
    "store_flights",
    "get_flights_from_db",
    "get_airport_codes",
    "match_airport",
    "format_airports_list",
    "get_unique_flight_routes",
    "get_flights_for_route",
]
