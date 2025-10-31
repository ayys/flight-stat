"""
Flight Status library for fetching and managing Buddha Air flight data.
"""

from flight_stat.lib import (
    AIRPORTS,
    analyze_existing_routes,
    close_database_pool,
    fetch_all_combinations_async,
    fetch_flight_status_async,
    format_airports_list,
    get_airport_codes,
    get_flights_for_route,
    get_flights_from_db,
    get_nepal_date,
    get_unique_flight_routes,
    init_database,
    match_airport,
    parse_xml,
    store_flights,
)

__all__ = [
    "AIRPORTS",
    "init_database",
    "close_database_pool",
    "fetch_flight_status_async",
    "fetch_all_combinations_async",
    "analyze_existing_routes",
    "get_nepal_date",
    "parse_xml",
    "store_flights",
    "get_flights_from_db",
    "get_airport_codes",
    "match_airport",
    "format_airports_list",
    "get_unique_flight_routes",
    "get_flights_for_route",
]
